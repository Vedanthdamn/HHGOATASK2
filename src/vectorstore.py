"""Vector store (Chroma, persistent) + BM25 keyword index for hybrid retrieval."""
import os
import pickle

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from src.chunking import Chunk
from src.config import config
from src.embeddings import Embedder
from src.fast_index import FastIndex

COLLECTION_NAME = "msmarco_xi_chunks"
BM25_PATH = os.path.join(config.CHROMA_PERSIST_DIR, "bm25.pkl")


def _tokenize(text: str) -> list[str]:
    """Index-side tokenizer. Unchanged: the persisted BM25 index was fitted
    with it, so altering it would invalidate the committed corpus."""
    return text.lower().split()


# Function words are stripped from BM25 *queries* (not from the index).
#
# BM25 divides term weight by document length, so a very short document that
# happens to contain a query's function words scores enormously on them. The
# corpus is full of 3-token question chunks ("वसेक्टॉमी क्या है?" -- "what is a
# vasectomy?"), and every query phrased as "X क्या है?" matched all of them on
# "क्या"/"है" alone. They then won rank 1 and the extractive tier returned one
# verbatim, so "फेनेल बीज क्या है?" ("what is fennel seed?") answered with
# "वसेक्टॉमी क्या है?" -- a different question, not even an answer.
#
# IDF alone does not save us here: rank_bm25 floors the IDF of very common
# terms at a small positive epsilon rather than letting it go negative, so
# function words keep contributing weight.
#
# Filtering at query time only, rather than re-fitting the index: measured on
# 200 queries this scored better on both answer quality (0.256 -> 0.269 cosine
# to MSMARCO's reference answers) and leakage-free retrieval (R@1 0.775 ->
# 0.795) than rebuilding the index with the same list, and it leaves the
# committed index untouched.
#
# Hindi + English, matching DATASET_LANG=hi. A different MSMARCO-XI language
# config would want its own list; an unrecognized language simply keeps all
# terms, i.e. today's behavior.
_QUERY_STOPWORDS = frozenset("""
क्या है हैं हूँ हूं का की के को में से पर और या यह वह ये वे कि जो तो ही भी था थी थे हो होता होती होते
कर करना किया एक इस उस अपने अपनी नहीं ना कैसे कब कहाँ कहां कौन क्यों किस किसी कोई कुछ मैं मेरा मेरी
हम आप सकते सकता सकती गया गई गए रहा रही रहे लिए लिये साथ द्वारा
what is are am the a an of in on to for and or how when where who why which do does did can could
will would i my we you it this that these those be been was were has have had with from by at as
""".split())

_STRIP_CHARS = "?।॥!.,:;\"'()[]{}"


def _tokenize_query(text: str) -> list[str]:
    """Index tokenization minus function words.

    Membership is tested against the punctuation-stripped form (index tokens
    carry trailing punctuation, e.g. "है?"), but the *original* token is kept
    so it still matches the index, which was built without stripping.
    """
    return [t for t in _tokenize(text) if t.strip(_STRIP_CHARS) not in _QUERY_STOPWORDS]


class VectorStore:
    def __init__(self, persist_dir: str = None, embedder: Embedder = None):
        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self.embedder = embedder or Embedder()
        self.bm25 = None
        self.bm25_chunk_ids: list[str] = []
        self._load_bm25()
        # Resident exact index for the query path. Built once here (a few
        # hundred ms) to take per-query SQLite round trips and full-corpus
        # Python scans off the hot path -- see src/fast_index.py. Best-effort:
        # if it can't be built, both legs fall back to the Chroma/rank_bm25
        # path below, which returns the same results more slowly.
        self.fast: FastIndex | None = None
        self._build_fast_index()

    def _build_fast_index(self):
        try:
            self.fast = FastIndex.from_store(self.collection, self.bm25, self.bm25_chunk_ids)
        except Exception:  # noqa: BLE001 - never let an optimization break startup
            self.fast = None

    # ---------------- indexing ----------------
    def add_chunks(self, chunks: list[Chunk], batch_size: int = 128):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.embedder.encode(texts)
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[
                    {
                        "strategy": c.strategy.value,
                        "doc_id": c.doc_id,
                        "query": c.query,
                        "answer": c.answer,
                        "is_selected": c.is_selected,
                        "query_id": c.query_id,
                        "query_type": c.query_type,
                        "lang": c.lang,
                    }
                    for c in batch
                ],
            )
        self._build_bm25(chunks, append=True)

    def _build_bm25(self, chunks: list[Chunk], append: bool = False):
        corpus_texts = [c.text for c in chunks]
        corpus_ids = [c.chunk_id for c in chunks]
        tokenized = [_tokenize(t) for t in corpus_texts]

        if append and self.bm25 is not None:
            existing_texts_path = os.path.join(config.CHROMA_PERSIST_DIR, "bm25_corpus.pkl")
            with open(existing_texts_path, "rb") as f:
                prev_tokenized, prev_ids = pickle.load(f)
            tokenized = prev_tokenized + tokenized
            corpus_ids = prev_ids + corpus_ids

        self.bm25 = BM25Okapi(tokenized)
        self.bm25_chunk_ids = corpus_ids
        with open(BM25_PATH, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(os.path.join(config.CHROMA_PERSIST_DIR, "bm25_corpus.pkl"), "wb") as f:
            pickle.dump((tokenized, corpus_ids), f)
        with open(os.path.join(config.CHROMA_PERSIST_DIR, "bm25_ids.pkl"), "wb") as f:
            pickle.dump(corpus_ids, f)

    def _load_bm25(self):
        ids_path = os.path.join(config.CHROMA_PERSIST_DIR, "bm25_ids.pkl")
        if os.path.exists(BM25_PATH) and os.path.exists(ids_path):
            with open(BM25_PATH, "rb") as f:
                self.bm25 = pickle.load(f)
            with open(ids_path, "rb") as f:
                self.bm25_chunk_ids = pickle.load(f)

    # ---------------- querying ----------------
    def semantic_search(self, query: str, top_k: int = None, query_embedding=None) -> list[dict]:
        top_k = top_k or config.TOP_K_SEMANTIC
        query_emb = query_embedding if query_embedding is not None else self.embedder.encode_one(query)
        if self.fast is not None:
            return self.fast.dense_search(query_emb, top_k)
        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
        )
        out = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            out.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - distance,  # cosine similarity
                    "source": "semantic",
                }
            )
        return out

    def bm25_search(self, query: str, top_k: int = None) -> list[dict]:
        top_k = top_k or config.TOP_K_BM25
        if self.bm25 is None:
            return []
        tokenized_query = _tokenize_query(query)
        if not tokenized_query:
            return []  # nothing but function words: no lexical signal, leave it to dense retrieval
        if self.fast is not None:
            return self.fast.bm25_search(tokenized_query, top_k)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        if not ranked:
            return []
        ids = [self.bm25_chunk_ids[i] for i in ranked]
        got = self.collection.get(ids=ids)
        by_id = {got["ids"][j]: (got["documents"][j], got["metadatas"][j]) for j in range(len(got["ids"]))}

        out = []
        max_score = max(scores[i] for i in ranked) or 1.0
        for i in ranked:
            cid = self.bm25_chunk_ids[i]
            if cid not in by_id:
                continue
            text, meta = by_id[cid]
            out.append(
                {
                    "chunk_id": cid,
                    "text": text,
                    "metadata": meta,
                    "score": float(scores[i]) / max_score,
                    "source": "bm25",
                }
            )
        return out

    def count(self) -> int:
        return self.collection.count()
