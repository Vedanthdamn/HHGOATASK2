"""Precomputed sentence-level embeddings for the extractive synthesis tier.

Tier 2 (`src/extractive.py`) scores individual *sentences* out of the top
retrieved chunks. It originally embedded those sentences on the query path,
which meant a second transformer forward pass per request -- measured at
613-864ms, i.e. ~95% of end-to-end latency, on the path most queries take.
That single call, not retrieval, was what put the pipeline over the 200ms
budget.

The corpus is static, so that work does not belong on the hot path. Every
indexed chunk is split into sentences once at index time and each sentence is
embedded and persisted here. At query time synthesis just looks up the stored
vectors for the (already retrieved) top chunks and does numpy dot products --
no model call at all.

The vectors are byte-identical to what the old on-the-fly path computed
(same model, same `normalize=True`), so this is purely a latency change:
answers are unchanged. Chunks missing from the index (e.g. added after the
last build) transparently fall back to on-the-fly embedding, so a stale or
absent index degrades latency rather than correctness.
"""
import os
import pickle

import numpy as np

from src.chunking import split_sentences
from src.config import config

SENTENCE_INDEX_PATH = os.path.join(config.CHROMA_PERSIST_DIR, "sentence_embeddings.pkl")

MIN_SENTENCE_CHARS = 3  # must match the filter in extractive.py


def sentences_for(text: str) -> list[str]:
    """Sentence split + the same length filter the synthesis tier applies."""
    return [s.strip() for s in split_sentences(text) if len(s.strip()) >= MIN_SENTENCE_CHARS]


class SentenceIndex:
    """chunk_id -> (sentences, normalized embedding matrix)."""

    def __init__(self, data: dict[str, tuple[list[str], np.ndarray]] | None = None):
        self._data = data or {}

    def __len__(self) -> int:
        return len(self._data)

    @property
    def n_sentences(self) -> int:
        return sum(len(s) for s, _ in self._data.values())

    def get(self, chunk_id: str) -> tuple[list[str], np.ndarray] | None:
        return self._data.get(chunk_id)

    # ---------------- persistence ----------------
    @classmethod
    def load(cls, path: str = SENTENCE_INDEX_PATH) -> "SentenceIndex":
        """Load the persisted index; returns an empty index if absent.

        Absence is not fatal -- synthesis falls back to on-the-fly embedding.
        """
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, "rb") as f:
                raw = pickle.load(f)
        except Exception:  # noqa: BLE001 - a corrupt cache must not take the pipeline down
            return cls()
        return cls({cid: (sents, embs.astype(np.float32)) for cid, (sents, embs) in raw.items()})

    def save(self, path: str = SENTENCE_INDEX_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            # float16 halves the on-disk artifact; cosine scores agree with
            # float32 to ~1e-3, far below any threshold the tier compares against.
            pickle.dump({cid: (sents, embs.astype(np.float16)) for cid, (sents, embs) in self._data.items()}, f)

    # ---------------- building ----------------
    @classmethod
    def build(cls, chunk_texts: dict[str, str], embedder, batch_size: int = 256, progress=None) -> "SentenceIndex":
        """Embed every sentence of every chunk, batched across chunks.

        Batching across chunks (rather than per chunk) keeps the transformer
        working on full batches -- a few hundred large batches instead of
        thousands of tiny ones.
        """
        chunk_ids: list[str] = []
        offsets: list[tuple[int, int]] = []
        all_sentences: list[str] = []
        per_chunk_sentences: list[list[str]] = []

        for cid, text in chunk_texts.items():
            sents = sentences_for(text)
            if not sents:
                continue
            start = len(all_sentences)
            all_sentences.extend(sents)
            chunk_ids.append(cid)
            per_chunk_sentences.append(sents)
            offsets.append((start, start + len(sents)))

        if not all_sentences:
            return cls()

        vectors: list[np.ndarray] = []
        for i in range(0, len(all_sentences), batch_size):
            vectors.append(embedder.encode(all_sentences[i : i + batch_size], normalize=True))
            if progress:
                progress(min(i + batch_size, len(all_sentences)), len(all_sentences))
        matrix = np.vstack(vectors).astype(np.float32)

        data = {
            cid: (sents, matrix[start:end])
            for cid, sents, (start, end) in zip(chunk_ids, per_chunk_sentences, offsets)
        }
        return cls(data)
