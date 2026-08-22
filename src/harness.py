"""Orchestration harness for the voice RAG pipeline.

Stages: STT -> input guardrail -> semantic cache check -> retrieval ->
retrieval guardrail -> context-injection guardrail -> generation (three
tiers, cheapest first) -> output guardrail -> structured result.

This is deliberately NOT a single prompt-in/text-out call: each stage has
its own typed input/output, its own error-recovery path, and its own
latency measurement, and any stage can short-circuit the pipeline with a
`refused` result instead of letting a failure or a bad answer propagate.

Generation is tiered cheapest-first, all the way down to a network call:
  1. extractive        - one retrieved chunk is an unambiguous match -> return it verbatim.
  2. extractive_synthesis - query-focused TextRank/LexRank across the top chunks -> stitch
                          together the most relevant sentences. Still zero network calls.
  3. llm                - Claude, only reached if neither non-LLM tier was confident enough.
Tiers 1-2 are why the *typical* query's full pipeline, not just the retrieval leg, lands
under 200ms -- the network call is the exception, not the default path.
"""
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from src import guardrails, stt
from src.embeddings import Embedder
from src.extractive import try_extractive_synthesis
from src.generation import generate_answer, try_extractive_answer
from src.retrieval import RetrievedChunk, hybrid_retrieve
from src.sentence_index import SentenceIndex
from src.vectorstore import VectorStore
from src.config import config


@dataclass
class StageTiming:
    stage: str
    ms: float


@dataclass
class PipelineResult:
    status: str  # "answered" | "refused" | "error"
    query: str
    answer: str = ""
    refusal_reason: str = ""
    refusal_stage: str = ""
    retrieved_chunks: list[dict] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    grounding_score: float | None = None
    retrieval_score: float | None = None
    timings_ms: list[dict] = field(default_factory=list)
    total_ms: float = 0.0
    retrieval_ms: float = 0.0  # chunking is precomputed at index time; this is the "vector DB retrieval" leg
    generation_mode: str = ""  # "extractive" | "extractive_synthesis" (no LLM call) | "llm" (Claude call)
    cache_hit: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class RagHarness:
    def __init__(self, store: VectorStore = None, embedder: Embedder = None, sentence_index: SentenceIndex = None):
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore(embedder=self.embedder)
        # Sentence vectors for the synthesis tier, precomputed at index time so
        # that tier costs numpy products instead of a transformer forward pass.
        # Loading is best-effort: an absent index just means the tier falls back
        # to embedding on the fly (slower, same answers).
        self.sentence_index = sentence_index if sentence_index is not None else SentenceIndex.load()
        self._cache: list[tuple[np.ndarray, PipelineResult]] = []

    def _time_stage(self, timings: list[StageTiming], name: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            return result, None
        except Exception as exc:  # noqa: BLE001 - deliberately broad: harness must not crash on stage failure
            return None, exc
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            timings.append(StageTiming(name, elapsed_ms))

    def _cache_lookup(self, query_embedding: np.ndarray) -> PipelineResult | None:
        best_sim, best_result = -1.0, None
        for cached_emb, cached_result in self._cache:
            sim = float(np.dot(query_embedding, cached_emb))
            if sim > best_sim:
                best_sim, best_result = sim, cached_result
        if best_result is not None and best_sim >= config.QUERY_CACHE_SIMILARITY_THRESHOLD:
            return best_result
        return None

    def _cache_store(self, query_embedding: np.ndarray, result: PipelineResult):
        if len(self._cache) >= config.QUERY_CACHE_MAX_SIZE:
            self._cache.pop(0)
        self._cache.append((query_embedding, result))

    def run(self, query: str | None = None, audio_bytes: bytes | None = None,
             language_code: str = "hi-IN", top_k: int = None, audio_filename: str = "audio.wav") -> PipelineResult:
        timings: list[StageTiming] = []
        t_start = time.perf_counter()

        # -- Stage 0: STT (only if audio was supplied instead of text) --
        if audio_bytes is not None:
            transcript_result, err = self._time_stage(timings, "stt", stt.transcribe_audio, audio_bytes, audio_filename, language_code)
            if err is not None:
                err = err.__cause__ or err
                return self._finish(PipelineResult(
                    status="error", query="", error=f"STT failed: {err}", refusal_stage="stt",
                ), timings, t_start)
            query = transcript_result["transcript"]

        if not query:
            return self._finish(PipelineResult(
                status="error", query="", error="No query text or audio provided.",
            ), timings, t_start)

        # -- Stage 1: input guardrail --
        input_check, err = self._time_stage(timings, "guardrail_input", guardrails.check_input_safety, query)
        if err is not None:
            return self._finish(PipelineResult(status="error", query=query, error=str(err)), timings, t_start)
        if not input_check.passed:
            return self._finish(PipelineResult(
                status="refused", query=query, refusal_reason=input_check.reason, refusal_stage="input",
            ), timings, t_start)

        # -- Stage 1.5: semantic cache lookup --
        # Embed once here and reuse the same vector for retrieval below, so the
        # cache check is ~free rather than a second embedding call.
        query_embedding, err = self._time_stage(timings, "embed_query", self.embedder.encode_one, query)
        if err is None and query_embedding is not None:
            cached, err2 = self._time_stage(timings, "cache_lookup", self._cache_lookup, query_embedding)
            if err2 is None and cached is not None:
                hit = PipelineResult(**{**cached.to_dict(), "cache_hit": True})
                return self._finish(hit, timings, t_start)

        # -- Stage 2: retrieval (vector DB + BM25 + fusion; embedding reused from stage 1.5) --
        retrieval_t0 = time.perf_counter()
        chunks, err = self._time_stage(timings, "retrieval", hybrid_retrieve, self.store, query, top_k, query_embedding)
        retrieval_ms = (time.perf_counter() - retrieval_t0) * 1000
        if err is not None:
            return self._finish(PipelineResult(status="error", query=query, error=str(err), retrieval_ms=retrieval_ms), timings, t_start)

        # -- Stage 3: retrieval-confidence guardrail --
        retrieval_check, err = self._time_stage(timings, "guardrail_retrieval", guardrails.check_retrieval_confidence, chunks)
        if err is not None:
            return self._finish(PipelineResult(status="error", query=query, error=str(err), retrieval_ms=retrieval_ms), timings, t_start)
        if not retrieval_check.passed:
            return self._finish(PipelineResult(
                status="refused", query=query, refusal_reason=retrieval_check.reason, refusal_stage="retrieval",
                retrieved_chunks=[asdict_chunk(c) for c in chunks], retrieval_score=retrieval_check.score,
                retrieval_ms=retrieval_ms,
            ), timings, t_start)

        # -- Stage 3.5: context-injection guardrail (scan retrieved passages themselves) --
        context_check, err = self._time_stage(timings, "guardrail_context", guardrails.check_context_injection, chunks)
        if err is not None:
            return self._finish(PipelineResult(status="error", query=query, error=str(err), retrieval_ms=retrieval_ms), timings, t_start)
        if not context_check.passed:
            return self._finish(PipelineResult(
                status="refused", query=query, refusal_reason=context_check.reason, refusal_stage="context",
                retrieved_chunks=[asdict_chunk(c) for c in chunks], retrieval_ms=retrieval_ms,
            ), timings, t_start)

        # -- Stage 4a: extractive fast path -- one chunk is an unambiguous match --
        gen_result, err = self._time_stage(timings, "generation_extractive", try_extractive_answer, chunks)
        if err is not None:
            gen_result = None

        # -- Stage 4b: extractive multi-chunk synthesis -- still zero network calls --
        if gen_result is None:
            gen_result, err = self._time_stage(
                timings, "generation_extractive_synthesis", try_extractive_synthesis, query, chunks, self.embedder,
                query_embedding, self.sentence_index,
            )
            if err is not None:
                gen_result = None

        # -- Stage 4c: LLM generation (structured tool-call, retried) -- last resort only --
        if gen_result is None:
            gen_result, err = self._time_stage(timings, "generation_llm", generate_answer, query, chunks)
            if err is not None:
                return self._finish(PipelineResult(
                    status="error", query=query, error=f"Generation failed: {err}",
                    retrieved_chunks=[asdict_chunk(c) for c in chunks], retrieval_ms=retrieval_ms,
                ), timings, t_start)

        if not gen_result.get("grounded", False) or not gen_result.get("answer", "").strip():
            return self._finish(PipelineResult(
                status="refused", query=query,
                refusal_reason=gen_result.get("refusal_reason") or "Model determined the context was insufficient.",
                refusal_stage="generation",
                retrieved_chunks=[asdict_chunk(c) for c in chunks], retrieval_ms=retrieval_ms,
                generation_mode=gen_result.get("mode", ""),
            ), timings, t_start)

        # -- Stage 5: output grounding guardrail (fast lexical token-overlap check) --
        output_check, err = self._time_stage(
            timings, "guardrail_output", guardrails.check_output_grounding_lexical, gen_result["answer"], chunks,
        )
        if err is not None:
            return self._finish(PipelineResult(
                status="error", query=query, error=str(err),
                retrieved_chunks=[asdict_chunk(c) for c in chunks], retrieval_ms=retrieval_ms,
            ), timings, t_start)

        if not output_check.passed:
            return self._finish(PipelineResult(
                status="refused", query=query, refusal_reason=output_check.reason, refusal_stage="output",
                retrieved_chunks=[asdict_chunk(c) for c in chunks], grounding_score=output_check.score,
                retrieval_ms=retrieval_ms, generation_mode=gen_result.get("mode", ""),
            ), timings, t_start)

        # -- Stage 5.5: coherence guardrail (catches degenerate/repetitive text,
        # e.g. an MT decoding-loop artifact -- would otherwise pass grounding
        # trivially since extractive answers are grounded in themselves) --
        coherence_check, err = self._time_stage(timings, "guardrail_coherence", guardrails.check_answer_coherence, gen_result["answer"])
        if err is not None:
            return self._finish(PipelineResult(
                status="error", query=query, error=str(err),
                retrieved_chunks=[asdict_chunk(c) for c in chunks], retrieval_ms=retrieval_ms,
            ), timings, t_start)
        if not coherence_check.passed:
            return self._finish(PipelineResult(
                status="refused", query=query, refusal_reason=coherence_check.reason, refusal_stage="coherence",
                retrieved_chunks=[asdict_chunk(c) for c in chunks], grounding_score=output_check.score,
                retrieval_ms=retrieval_ms, generation_mode=gen_result.get("mode", ""),
            ), timings, t_start)

        result = self._finish(PipelineResult(
            status="answered", query=query, answer=gen_result["answer"],
            cited_chunk_ids=gen_result.get("cited_chunk_ids", []),
            retrieved_chunks=[asdict_chunk(c) for c in chunks],
            grounding_score=output_check.score, retrieval_score=retrieval_check.score,
            retrieval_ms=retrieval_ms, generation_mode=gen_result.get("mode", ""),
        ), timings, t_start)

        if query_embedding is not None:
            self._cache_store(query_embedding, result)
        return result

    def _finish(self, result: PipelineResult, timings: list[StageTiming], t_start: float) -> PipelineResult:
        result.timings_ms = [asdict(t) for t in timings]
        result.total_ms = (time.perf_counter() - t_start) * 1000
        return result


def asdict_chunk(c: RetrievedChunk) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "text": c.text,
        "score": c.score,
        "metadata": c.metadata,
        "sources": c.sources,
    }
