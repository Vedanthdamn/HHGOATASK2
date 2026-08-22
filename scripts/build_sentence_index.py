"""Precompute sentence-level embeddings for the extractive synthesis tier.

Reads the chunks already in the Chroma collection (no dataset download
needed) and writes data/chroma/sentence_embeddings.pkl. Run this after
scripts/build_index.py, and again any time the chunk index is rebuilt.

Usage:
    python scripts/build_sentence_index.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import Embedder
from src.sentence_index import SENTENCE_INDEX_PATH, SentenceIndex
from src.vectorstore import VectorStore


def main():
    embedder = Embedder()
    store = VectorStore(embedder=embedder)
    print(f"Vector store has {store.count()} indexed chunks.")

    got = store.collection.get(include=["documents"])
    chunk_texts = {cid: doc for cid, doc in zip(got["ids"], got["documents"]) if doc}
    print(f"Splitting and embedding sentences for {len(chunk_texts)} chunks ...")

    t0 = time.perf_counter()

    def progress(done, total):
        print(f"  {done}/{total} sentences embedded", end="\r", flush=True)

    index = SentenceIndex.build(chunk_texts, embedder, progress=progress)
    elapsed = time.perf_counter() - t0

    index.save()
    size_mb = Path(SENTENCE_INDEX_PATH).stat().st_size / 1e6
    print(
        f"\nIndexed {index.n_sentences} sentences across {len(index)} chunks "
        f"in {elapsed:.1f}s -> {SENTENCE_INDEX_PATH} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
