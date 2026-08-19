"""Orchestration harness for the voice RAG pipeline.

Stages: STT -> input guardrail -> retrieval -> retrieval guardrail ->
generation (structured tool call) -> output guardrail -> structured result.

This is deliberately NOT a single prompt-in/text-out call: each stage has
its own typed input/output, its own error-recovery path, and its own
latency measurement, and any stage can short-circuit the pipeline with a
`refused` result instead of letting a failure or a bad answer propagate.
"""
import time
import traceback
from dataclasses import asdict, dataclass, field

from src import guardrails, stt
from src.config import config
from src.embeddings import Embedder
from src.generation import generate_answer
from src.retrieval import RetrievedChunk, hybrid_retrieve
from src.vectorstore import VectorStore


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
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class RagHarness:
    def __init__(self, store: VectorStore = None, embedder: Embedder = None):
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore(embedder=self.embedder)

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

    def run(self, query: str | None = None, audio_bytes: bytes | None = None,
             language_code: str = "hi-IN", top_k: int = None) -> PipelineResult:
        timings: list[StageTiming] = []
        t_start = time.perf_counter()

        # -- Stage 0: STT (only if audio was supplied instead of text) --
        if audio_bytes is not None:
            transcript_result, err = self._time_stage(timings, "stt", stt.transcribe_audio, audio_bytes, "audio.wav", language_code)
            if err is not None:
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

        # -- Stage 2: retrieval (embedding + vector DB + BM25 + fusion) --
        retrieval_t0 = time.perf_counter()
        chunks, err = self._time_stage(timings, "retrieval", hybrid_retrieve, self.store, query, top_k)
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

        # -- Stage 4: generation (structured tool-call, retried) --
        gen_result, err = self._time_stage(timings, "generation", generate_answer, query, chunks)
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
            ), timings, t_start)

        # -- Stage 5: output grounding guardrail (independent, embedding-based check) --
        output_check, err = self._time_stage(
            timings, "guardrail_output", guardrails.check_output_grounding, gen_result["answer"], chunks, self.embedder,
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
                retrieval_ms=retrieval_ms,
            ), timings, t_start)

        return self._finish(PipelineResult(
            status="answered", query=query, answer=gen_result["answer"],
            cited_chunk_ids=gen_result.get("cited_chunk_ids", []),
            retrieved_chunks=[asdict_chunk(c) for c in chunks],
            grounding_score=output_check.score, retrieval_score=retrieval_check.score,
            retrieval_ms=retrieval_ms,
        ), timings, t_start)

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
