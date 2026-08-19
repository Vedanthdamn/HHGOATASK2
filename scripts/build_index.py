"""Downloads a sample of MSMARCO-XI, chunks it with the strategy router, and
indexes the chunks into the persistent Chroma + BM25 store.

Usage:
    python scripts/build_index.py --lang hi --sample-size 2000
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_documents
from src.config import config
from src.dataset import load_msmarco_documents
from src.embeddings import Embedder
from src.vectorstore import VectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=config.DATASET_LANG)
    parser.add_argument("--split", default=config.DATASET_SPLIT)
    parser.add_argument("--sample-size", type=int, default=config.DATASET_SAMPLE_SIZE)
    args = parser.parse_args()

    print(f"Loading MSMARCO-XI[{args.lang}] split={args.split} sample_size={args.sample_size} ...")
    t0 = time.perf_counter()
    docs = load_msmarco_documents(lang=args.lang, split=args.split, sample_size=args.sample_size)
    print(f"Loaded {len(docs)} passage-documents in {time.perf_counter() - t0:.1f}s")

    print("Loading embedder (used for semantic chunking breakpoints too)...")
    embedder = Embedder()

    print("Chunking with strategy router (atomic / semantic / sentence-window / fixed-size)...")
    t0 = time.perf_counter()
    chunks = chunk_documents(docs, embedder=embedder)
    print(f"Produced {len(chunks)} chunks from {len(docs)} documents in {time.perf_counter() - t0:.1f}s")

    strategy_counts = {}
    for c in chunks:
        strategy_counts[c.strategy.value] = strategy_counts.get(c.strategy.value, 0) + 1
    print("Chunking strategy breakdown:", strategy_counts)

    print(f"Indexing into Chroma + BM25 at {config.CHROMA_PERSIST_DIR} ...")
    t0 = time.perf_counter()
    store = VectorStore(embedder=embedder)
    store.add_chunks(chunks)
    print(f"Indexed {store.count()} total chunks in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
