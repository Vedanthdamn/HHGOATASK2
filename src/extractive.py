"""Non-LLM multi-chunk extractive synthesis (query-focused TextRank/LexRank).

`try_extractive_answer` (in generation.py) only fires when one retrieved
chunk is an unambiguous winner. Most "needs synthesis across chunks"
queries used to fall through to a multi-second Claude call for that reason
alone -- not because they were actually hard, just because the answer
wasn't sitting verbatim in a single chunk.

This module is the second, still-zero-network-call tier: pull sentences out
of the top retrieved chunks, score each by a blend of (a) similarity to the
query and (b) centrality in the sentence-similarity graph (power iteration,
i.e. TextRank/LexRank), keep the best few, and stitch them back together in
their original order. It costs one batched embedding call over a couple
dozen short sentences (a few ms) instead of a network round-trip.
"""
import numpy as np

from src.chunking import split_sentences
from src.config import config
from src.embeddings import Embedder
from src.retrieval import RetrievedChunk

QUERY_RELEVANCE_WEIGHT = 0.65  # vs. (1 - this) weight on graph centrality
MIN_SYNTHESIS_CONFIDENCE = 0.30  # avg query-relevance of selected sentences; below this, defer to the LLM
MAX_SENTENCES = 3
MAX_CHUNKS_CONSIDERED = 5
DAMPING = 0.85
POWER_ITERATIONS = 20


def _centrality_scores(sentence_embeddings: np.ndarray) -> np.ndarray:
    n = len(sentence_embeddings)
    if n == 1:
        return np.array([1.0])

    norm = sentence_embeddings / (np.linalg.norm(sentence_embeddings, axis=1, keepdims=True) + 1e-8)
    sim = norm @ norm.T
    np.fill_diagonal(sim, 0.0)
    sim = np.clip(sim, 0.0, None)

    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = sim / row_sums

    scores = np.full(n, 1.0 / n)
    for _ in range(POWER_ITERATIONS):
        scores = (1 - DAMPING) / n + DAMPING * (transition.T @ scores)
    return scores


def try_extractive_synthesis(query: str, chunks: list[RetrievedChunk], embedder: Embedder,
                             query_embedding=None, sentence_index=None) -> dict | None:
    candidate_chunks = chunks[:MAX_CHUNKS_CONSIDERED]

    sentences: list[str] = []
    sentence_chunk_ids: list[str] = []
    sentence_order: list[tuple[int, int]] = []  # (chunk_rank, sentence_index) for stitching back in order
    # Sentence vectors are assembled per chunk: precomputed where the chunk is
    # in the index, embedded on the fly only for the ones that aren't. Each
    # entry below is either a stored vector block or a (start, end) slice of
    # `sentences` still awaiting embedding.
    #
    # The fallback is deliberately per-chunk rather than all-or-nothing: an
    # earlier version bailed to embedding *everything* if any one chunk was
    # missing, so a single unindexed chunk cost 193ms instead of 0.4ms and
    # single-handedly blew the latency budget (see README).
    blocks: list[np.ndarray | tuple[int, int]] = []

    for chunk_rank, c in enumerate(candidate_chunks):
        stored = sentence_index.get(c.chunk_id) if sentence_index is not None else None
        if stored is not None:
            chunk_sentences, chunk_embs = stored
        else:
            chunk_sentences = [s.strip() for s in split_sentences(c.text) if len(s.strip()) >= 3]
            chunk_embs = None
        if not chunk_sentences:
            continue  # nothing to contribute; costs nothing either way

        start = len(sentences)
        for sent_idx, sent in enumerate(chunk_sentences):
            sentences.append(sent)
            sentence_chunk_ids.append(c.chunk_id)
            sentence_order.append((chunk_rank, sent_idx))
        blocks.append(chunk_embs if chunk_embs is not None else (start, len(sentences)))

    if len(sentences) < 2:
        return None  # not enough material to synthesize; let another tier handle it

    query_emb = np.asarray(query_embedding, dtype=np.float32) if query_embedding is not None else None
    pending = [b for b in blocks if isinstance(b, tuple)]

    if query_emb is None or pending:
        # A single batched call covering only what is genuinely missing: the
        # query (if it wasn't already embedded upstream for retrieval) plus the
        # sentences of any unindexed chunks.
        to_embed = ([] if query_emb is not None else [query])
        for start, end in pending:
            to_embed.extend(sentences[start:end])
        fresh = embedder.encode(to_embed, normalize=True)
        if query_emb is None:
            query_emb, fresh = fresh[0], fresh[1:]
        cursor = 0
        for i, b in enumerate(blocks):
            if isinstance(b, tuple):
                n = b[1] - b[0]
                blocks[i] = fresh[cursor : cursor + n]
                cursor += n

    sentence_embs = np.vstack(blocks).astype(np.float32)

    # Apple Accelerate's BLAS on macOS/arm64 raises spurious invalid/overflow
    # RuntimeWarnings on some matmul shapes even though the result is exact
    # (verified: no NaN/Inf in inputs or outputs) -- silence locally rather
    # than let it mask a real warning elsewhere in the pipeline.
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        query_relevance = sentence_embs @ query_emb  # already normalized -> cosine sim
        centrality = _centrality_scores(sentence_embs)
    # min-max normalize centrality onto the same ~[0,1] range as cosine sim before blending
    c_range = centrality.max() - centrality.min()
    centrality_norm = (centrality - centrality.min()) / c_range if c_range > 1e-8 else centrality

    combined = QUERY_RELEVANCE_WEIGHT * query_relevance + (1 - QUERY_RELEVANCE_WEIGHT) * centrality_norm

    top_n = min(MAX_SENTENCES, len(sentences))
    top_indices = np.argsort(combined)[::-1][:top_n]

    avg_confidence = float(query_relevance[top_indices].mean())
    if avg_confidence < MIN_SYNTHESIS_CONFIDENCE:
        return None  # selected sentences aren't actually close to the query; defer to the LLM

    ordered_indices = sorted(top_indices.tolist(), key=lambda i: sentence_order[i])
    answer_text = " ".join(sentences[i] for i in ordered_indices)
    cited = sorted({sentence_chunk_ids[i] for i in ordered_indices})

    return {
        "grounded": True,
        "answer": answer_text,
        "cited_chunk_ids": cited,
        "refusal_reason": None,
        "mode": "extractive_synthesis",
    }
