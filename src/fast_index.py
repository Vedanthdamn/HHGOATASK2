"""In-memory exact search index for the retrieval hot path.

Retrieval was measured at ~90ms on the deployed instance for a 3,281-chunk
corpus, which is far more than the work actually requires: a 3,281 x 384
dense scan is ~1.3M multiply-adds, i.e. microseconds. The time was going to
per-query *overhead* rather than to search:

  1. `chroma.collection.query()` -- HNSW plus a SQLite read plus marshalling
     Python objects, per query.
  2. `rank_bm25.BM25Okapi.get_scores()` -- builds a full-corpus numpy array
     per query term via a Python list comprehension over every document
     (`[doc.get(q) or 0 for doc in self.doc_freqs]`), so cost scales with
     corpus size even though only the handful of documents containing the
     term can score above zero.
  3. `sorted(range(n))` over every document to take the top 8.
  4. A *second* SQLite round trip (`collection.get(ids=...)`) purely to fetch
     the text and metadata of the BM25 hits.

At this corpus size all four are pure overhead, so this module keeps the
whole corpus resident and answers both legs from memory:

  - **Dense**: one normalized `(n_chunks, dim)` float32 matrix. `M @ q` is a
    single BLAS call and, unlike HNSW, is *exact* -- there is no recall
    cliff to tune around.
  - **Sparse**: a classic inverted index (term -> document ids + precomputed
    BM25 weights). Cost scales with postings actually touched, not corpus
    size. The weights are derived from the already-fitted `BM25Okapi`
    object, so scores are identical to what it produces rather than merely
    similar (see `_build_postings`).
  - **Documents**: `chunk_id -> (text, metadata)` in a dict, so neither leg
    needs SQLite at query time.

Built once at startup from the persisted Chroma collection and BM25 pickle.
Deliberately *not* a new on-disk artifact: it is derived state, and a cached
copy could silently disagree with the index it was built from.
"""
import numpy as np


class FastIndex:
    def __init__(self, chunk_ids, matrix, docs, postings, idf_default=0.0):
        self.chunk_ids: list[str] = chunk_ids
        self.matrix: np.ndarray = matrix  # (n, dim), L2-normalized float32
        self.docs: dict[str, tuple[str, dict]] = docs
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = postings
        self._idf_default = idf_default
        self._n = len(chunk_ids)

    def __len__(self) -> int:
        return self._n

    # ---------------- construction ----------------
    @classmethod
    def from_store(cls, collection, bm25, bm25_chunk_ids) -> "FastIndex | None":
        """Materialize from a Chroma collection + a fitted BM25Okapi.

        Returns None if the collection is empty or embeddings are unavailable,
        so callers can fall back to the original query path.
        """
        got = collection.get(include=["embeddings", "documents", "metadatas"])
        ids = got.get("ids") or []
        embeddings = got.get("embeddings")
        if not ids or embeddings is None or len(embeddings) == 0:
            return None

        matrix = np.asarray(embeddings, dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8

        documents = got.get("documents") or []
        metadatas = got.get("metadatas") or []
        docs = {
            cid: (documents[i] if i < len(documents) else "",
                  metadatas[i] if i < len(metadatas) else {})
            for i, cid in enumerate(ids)
        }

        postings = cls._build_postings(bm25, bm25_chunk_ids, ids) if bm25 is not None else {}
        return cls(list(ids), matrix, docs, postings)

    @staticmethod
    def _build_postings(bm25, bm25_chunk_ids, dense_ids) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Precompute each (term, document) BM25 contribution.

        BM25Okapi scores a document as

            sum over query terms q of
                idf[q] * f(q,d)*(k1+1) / (f(q,d) + k1*(1 - b + b*len(d)/avgdl))

        Every factor except the choice of q depends only on (term, document),
        so the whole per-pair value is precomputable. Query time then reduces
        to summing the postings of the query's terms.

        Values are read off the already-fitted BM25Okapi (its own idf, k1, b,
        avgdl and per-document term frequencies), so this reproduces its
        scores exactly instead of reimplementing the formula and hoping the
        constants match. Documents are keyed by chunk id and remapped onto the
        dense matrix's row order so both legs share one index space.
        """
        row_of = {cid: i for i, cid in enumerate(dense_ids)}
        k1, b, avgdl = bm25.k1, bm25.b, bm25.avgdl
        doc_len = bm25.doc_len

        acc: dict[str, tuple[list[int], list[float]]] = {}
        for doc_i, freqs in enumerate(bm25.doc_freqs):
            if doc_i >= len(bm25_chunk_ids):
                break
            row = row_of.get(bm25_chunk_ids[doc_i])
            if row is None:
                continue  # indexed by BM25 but absent from the dense collection
            denom_norm = k1 * (1 - b + b * doc_len[doc_i] / avgdl)
            for term, f in freqs.items():
                idf = bm25.idf.get(term)
                if not idf:
                    continue
                weight = idf * (f * (k1 + 1)) / (f + denom_norm)
                rows, weights = acc.setdefault(term, ([], []))
                rows.append(row)
                weights.append(weight)

        return {
            term: (np.asarray(rows, dtype=np.int32), np.asarray(weights, dtype=np.float32))
            for term, (rows, weights) in acc.items()
        }

    # ---------------- querying ----------------
    def dense_search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        q = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        scores = self.matrix @ q  # cosine similarity; both sides normalized
        return self._top(scores, top_k, "semantic", normalize_by_max=False)

    def bm25_search(self, tokens: list[str], top_k: int) -> list[dict]:
        if not self._postings:
            return []
        scores = np.zeros(self._n, dtype=np.float32)
        hit = False
        for term in tokens:  # per occurrence, matching BM25Okapi's own loop
            posting = self._postings.get(term)
            if posting is None:
                continue
            rows, weights = posting
            # Plain fancy-index accumulation rather than np.add.at (much
            # faster): a term appears at most once per document in
            # `doc_freqs`, so a single posting list never repeats a row.
            scores[rows] += weights
            hit = True
        if not hit:
            return []
        return self._top(scores, top_k, "bm25", normalize_by_max=True)

    def _top(self, scores: np.ndarray, top_k: int, source: str, normalize_by_max: bool) -> list[dict]:
        k = min(top_k, self._n)
        if k <= 0:
            return []
        # A *stable* full sort, not argpartition + unstable argsort. Ties are
        # common here (BM25 in particular produces many equal scores), and
        # Python's `sorted(range(n), key=..., reverse=True)` -- what this
        # replaces -- is stable, so equal scores keep ascending index order.
        # An unstable sort reorders those ties arbitrarily, which silently
        # changed which chunks were retrieved. Sorting 3k floats costs ~0.1ms,
        # cheap enough not to trade determinism for it.
        idx = np.argsort(-scores, kind="stable")[:k]

        # BM25 scores are reported relative to the best hit in the returned
        # set, as the original implementation did.
        denom = float(scores[idx[0]]) if normalize_by_max else 1.0
        if not denom:
            denom = 1.0

        out = []
        for i in idx:
            cid = self.chunk_ids[i]
            text, meta = self.docs.get(cid, ("", {}))
            out.append({
                "chunk_id": cid,
                "text": text,
                "metadata": meta,
                "score": float(scores[i]) / denom,
                "source": source,
            })
        return out
