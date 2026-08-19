"""Hybrid retrieval: semantic (dense) + BM25 (sparse), fused with Reciprocal
Rank Fusion, then re-ranked with a metadata-aware boost for chunks whose
source passage was marked `is_selected=True` in the original MSMARCO
annotations (a cheap, free relevance signal already present in the dataset).
"""
from dataclasses import dataclass

from src.config import config
from src.vectorstore import VectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict
    sources: list[str]


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = None) -> dict[str, float]:
    k = k or config.RRF_K
    fused: dict[str, float] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            fused[item["chunk_id"]] = fused.get(item["chunk_id"], 0.0) + 1.0 / (k + rank + 1)
    return fused


def hybrid_retrieve(store: VectorStore, query: str, top_k: int = None) -> list[RetrievedChunk]:
    top_k = top_k or config.TOP_K_FINAL

    semantic_results = store.semantic_search(query)
    bm25_results = store.bm25_search(query)

    by_id: dict[str, dict] = {}
    sources: dict[str, set[str]] = {}
    for results in (semantic_results, bm25_results):
        for r in results:
            by_id[r["chunk_id"]] = r
            sources.setdefault(r["chunk_id"], set()).add(r["source"])

    fused_scores = reciprocal_rank_fusion([semantic_results, bm25_results])

    # metadata-aware boost: chunks whose passage was flagged relevant
    # in the dataset's own annotations get a small score bump
    SELECTED_BOOST = 0.15
    for cid, item in by_id.items():
        if item["metadata"].get("is_selected"):
            fused_scores[cid] = fused_scores.get(cid, 0.0) + SELECTED_BOOST

    ranked_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

    max_score = max(fused_scores.values()) if fused_scores else 1.0
    out = []
    for cid in ranked_ids:
        item = by_id[cid]
        out.append(
            RetrievedChunk(
                chunk_id=cid,
                text=item["text"],
                score=fused_scores[cid] / max_score if max_score else 0.0,
                metadata=item["metadata"],
                sources=sorted(sources[cid]),
            )
        )
    return out
