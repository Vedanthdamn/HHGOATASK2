"""Chunking strategies for the MSMARCO-XI passage corpus.

MSMARCO passages are mostly short (1-4 sentences), so a single fixed-size
splitter would either no-op on most of the corpus or shred the few long
passages arbitrarily. Instead we run a small strategy router:

  - ATOMIC:            passage is already short -> keep as one chunk
                        (avoids destroying tiny bits of context).
  - SENTENCE_WINDOW:    sentence-level split with a sliding overlap window
                        (good recall for procedural/list-like passages).
  - SEMANTIC:           breakpoint-based split using embedding similarity
                        between consecutive sentences (keeps topically
                        coherent spans together instead of cutting at a
                        fixed character/token boundary).
  - FIXED_SIZE:         classic fixed-size char window with overlap, used
                        as a fallback / baseline and for anything the other
                        strategies can't confidently segment.

Every chunk carries metadata-aware tags (query_id, query_type, is_selected,
lang, strategy) so retrieval and guardrails can reason about provenance.
"""
import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.dataset import Document


class ChunkStrategy(str, Enum):
    ATOMIC = "atomic"
    SENTENCE_WINDOW = "sentence_window"
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    strategy: ChunkStrategy
    doc_id: str
    query: str
    answer: str
    is_selected: bool
    query_id: str
    query_type: str
    lang: str
    metadata: dict = field(default_factory=dict)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+")


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or [text.strip()]


def _make_chunk(doc: Document, text: str, idx: int, strategy: ChunkStrategy, extra: dict) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}-{strategy.value}-{idx}",
        text=text,
        strategy=strategy,
        doc_id=doc.doc_id,
        query=doc.query,
        answer=doc.answer,
        is_selected=doc.is_selected,
        query_id=doc.query_id,
        query_type=doc.query_type,
        lang=doc.lang,
        metadata={**doc.metadata, **extra},
    )


def chunk_atomic(doc: Document) -> list[Chunk]:
    return [_make_chunk(doc, doc.text, 0, ChunkStrategy.ATOMIC, {})]


def chunk_fixed_size(doc: Document, size: int = 400, overlap: int = 80) -> list[Chunk]:
    text = doc.text
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(_make_chunk(doc, piece, idx, ChunkStrategy.FIXED_SIZE, {"start": start, "end": end}))
            idx += 1
        if end == len(text):
            break
        start = end - overlap
    return chunks


def chunk_sentence_window(doc: Document, window: int = 3, overlap: int = 1) -> list[Chunk]:
    sentences = split_sentences(doc.text)
    if len(sentences) <= window:
        return [_make_chunk(doc, doc.text, 0, ChunkStrategy.SENTENCE_WINDOW, {"n_sentences": len(sentences)})]

    chunks = []
    idx = 0
    step = max(window - overlap, 1)
    i = 0
    while i < len(sentences):
        window_sents = sentences[i : i + window]
        if not window_sents:
            break
        piece = " ".join(window_sents)
        chunks.append(
            _make_chunk(
                doc, piece, idx, ChunkStrategy.SENTENCE_WINDOW,
                {"sent_start": i, "sent_end": i + len(window_sents)},
            )
        )
        idx += 1
        if i + window >= len(sentences):
            break
        i += step
    return chunks


def chunk_semantic(doc: Document, embedder, percentile_threshold: float = 90.0) -> list[Chunk]:
    """Breakpoint-based semantic chunking (TextTiling-style).

    Embeds each sentence, computes cosine distance between consecutive
    sentences, and cuts a new chunk wherever that distance exceeds the given
    percentile of the document's own distance distribution -> keeps
    topically coherent runs of sentences together instead of a blind
    fixed-size cut.
    """
    sentences = split_sentences(doc.text)
    if len(sentences) <= 2:
        return chunk_atomic(doc)

    embeddings = embedder.encode(sentences, normalize=True)
    dists = []
    for i in range(len(embeddings) - 1):
        cos_sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        dists.append(1.0 - cos_sim)

    if not dists:
        return chunk_atomic(doc)

    threshold = float(np.percentile(dists, percentile_threshold))
    breakpoints = {i + 1 for i, d in enumerate(dists) if d > threshold}

    chunks = []
    current = [sentences[0]]
    idx = 0
    for i in range(1, len(sentences)):
        if i in breakpoints:
            piece = " ".join(current)
            chunks.append(_make_chunk(doc, piece, idx, ChunkStrategy.SEMANTIC, {"breakpoint_dist_threshold": threshold}))
            idx += 1
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        piece = " ".join(current)
        chunks.append(_make_chunk(doc, piece, idx, ChunkStrategy.SEMANTIC, {"breakpoint_dist_threshold": threshold}))
    return chunks


def chunk_document(doc: Document, embedder=None, short_char_threshold: int = 220) -> list[Chunk]:
    """Router: pick a strategy based on passage shape.

    - very short passages -> ATOMIC (don't shred a 1-sentence passage)
    - passages with clear sentence boundaries and moderate length -> SEMANTIC
      (falls back to SENTENCE_WINDOW if no embedder was supplied)
    - anything else -> FIXED_SIZE as a robust fallback
    """
    text = doc.text
    if len(text) <= short_char_threshold:
        return chunk_atomic(doc)

    sentences = split_sentences(text)
    if len(sentences) >= 3:
        if embedder is not None:
            return chunk_semantic(doc, embedder)
        return chunk_sentence_window(doc)

    return chunk_fixed_size(doc)


def chunk_documents(docs: list[Document], embedder=None) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, embedder=embedder))
    return all_chunks
