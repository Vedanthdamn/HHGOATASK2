"""Guardrails: input safety/relevance checks, retrieval-confidence gating,
context-injection scanning, and post-hoc grounding verification.

Four independent layers, each of which can veto the pipeline before it
returns an answer to the user:

  1. INPUT     - blocks unsafe content and obviously off-topic chit-chat
                 before we spend a retrieval + generation call on it.
  2. RETRIEVAL - if nothing in the corpus is actually relevant (low fused
                 score), refuse rather than let the LLM improvise an answer.
  3. CONTEXT   - scans retrieved chunk text itself for injection/unsafe
                 patterns before it reaches the generator's prompt (MSMARCO
                 passages are scraped web text, so this is a real surface,
                 not a hypothetical one).
  4. OUTPUT    - compares the answer's tokens against the retrieved context;
                 if the answer contains claims not traceable to any
                 retrieved chunk, treat it as ungrounded/hallucinated and
                 refuse instead of returning it. Lexical (token-overlap)
                 rather than embedding-based on purpose: it is a pure string
                 operation with no model forward pass, so it costs
                 microseconds instead of ~100ms per request -- the
                 embedding-based version is kept as `check_output_grounding`
                 for callers that want a semantic (paraphrase-tolerant)
                 check and can afford the latency.
"""
import re
from dataclasses import dataclass

import numpy as np

from src.config import config
from src.embeddings import Embedder
from src.retrieval import RetrievedChunk

_UNSAFE_PATTERNS = [
    r"\bbomb\b", r"\bexplosive\b", r"\bkill (myself|yourself)\b",
    r"\bsuicide\b", r"\bself[- ]harm\b", r"\bhow to (make|build) (a )?(weapon|gun|bioweapon)\b",
    r"\bchild (sexual|porn)\b", r"\bcsam\b",
]
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)

_INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"disregard (all |the )?(previous|prior|above) instructions",
    r"reveal (your |the )?(system prompt|instructions)",
    r"you are now (in )?(dan|developer mode|jailbreak)",
    r"act as (if you (are|were)|an unrestricted)",
    r"\bsystem prompt\b.*\b(override|bypass|ignore)\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MIN_QUERY_LEN = 3


@dataclass
class GuardrailResult:
    passed: bool
    stage: str
    reason: str = ""
    score: float | None = None


def check_input_safety(query: str) -> GuardrailResult:
    q = (query or "").strip()
    if len(q) < _MIN_QUERY_LEN:
        return GuardrailResult(False, "input", "Query is too short/empty to answer.")
    if _UNSAFE_RE.search(q):
        return GuardrailResult(False, "input", "Query matched an unsafe-content pattern.")
    return GuardrailResult(True, "input")


def check_retrieval_confidence(chunks: list[RetrievedChunk]) -> GuardrailResult:
    if not chunks:
        return GuardrailResult(False, "retrieval", "No chunks retrieved from the index.")

    top_score = max(c.score for c in chunks)
    if top_score < config.MIN_RETRIEVAL_SCORE:
        return GuardrailResult(
            False, "retrieval",
            f"Top retrieval score {top_score:.2f} is below the confidence floor "
            f"({config.MIN_RETRIEVAL_SCORE}); the corpus likely has no relevant passage.",
            score=top_score,
        )

    # `score` is rank-fused then normalized by the candidate set's own max,
    # so the top result is *always* ~1.0 even when nothing in the corpus is
    # actually relevant to an off-topic query -- it's a relative "best of
    # this pool" signal, not an absolute one. raw_semantic_score (un-fused
    # cosine similarity) doesn't have that blind spot: verified empirically
    # that genuinely on-topic queries score 0.60-0.82 while an off-topic
    # query topped out at 0.47 against this corpus -- so this catches cases
    # the relative score alone cannot.
    top_raw = max(c.raw_semantic_score for c in chunks)
    if top_raw < config.MIN_RAW_SEMANTIC_SCORE:
        return GuardrailResult(
            False, "retrieval",
            f"Top raw semantic similarity {top_raw:.2f} is below the absolute confidence floor "
            f"({config.MIN_RAW_SEMANTIC_SCORE}); query is likely off-topic for this corpus.",
            score=top_raw,
        )

    return GuardrailResult(True, "retrieval", score=top_score)


def check_context_injection(chunks: list[RetrievedChunk]) -> GuardrailResult:
    """Scan retrieved passages themselves for prompt-injection attempts
    before they're concatenated into the generator's context window.
    Indirect prompt injection (poisoned corpus text instructing the model to
    ignore its system prompt) is a documented RAG attack surface distinct
    from a malicious user query.
    """
    for c in chunks:
        if _INJECTION_RE.search(c.text) or _UNSAFE_RE.search(c.text):
            return GuardrailResult(
                False, "context",
                f"Retrieved chunk {c.chunk_id} matched an injection/unsafe pattern; withholding it from generation.",
            )
    return GuardrailResult(True, "context")


def _tokenize_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def check_output_grounding_lexical(answer: str, chunks: list[RetrievedChunk]) -> GuardrailResult:
    """Fast path: fraction of the answer's distinct tokens that also appear
    somewhere in the retrieved context. Pure set arithmetic over already-
    in-memory strings, so it costs microseconds -- no model call, unlike
    `check_output_grounding` below. This is the default output guardrail;
    the embedding version remains available for callers that need
    paraphrase tolerance and can spend the latency on it.
    """
    if not answer or not answer.strip():
        return GuardrailResult(False, "output", "Empty answer generated.")
    if not chunks:
        return GuardrailResult(False, "output", "No context to ground the answer against.")

    answer_words = _tokenize_words(answer)
    if not answer_words:
        return GuardrailResult(False, "output", "Answer had no extractable tokens.")

    context_words: set[str] = set()
    for c in chunks:
        context_words |= _tokenize_words(c.text)

    overlap = len(answer_words & context_words) / len(answer_words)
    if overlap < config.MIN_LEXICAL_GROUNDING_OVERLAP:
        return GuardrailResult(
            False, "output",
            f"Answer/context token overlap {overlap:.2f} is below the grounding floor "
            f"({config.MIN_LEXICAL_GROUNDING_OVERLAP}); answer may not be grounded in retrieved context.",
            score=overlap,
        )
    return GuardrailResult(True, "output", score=overlap)


def check_output_grounding(answer: str, chunks: list[RetrievedChunk], embedder: Embedder) -> GuardrailResult:
    if not answer or not answer.strip():
        return GuardrailResult(False, "output", "Empty answer generated.")
    if not chunks:
        return GuardrailResult(False, "output", "No context to ground the answer against.")

    context_text = " ".join(c.text for c in chunks)
    answer_emb = embedder.encode_one(answer)
    context_emb = embedder.encode_one(context_text)

    sim = float(np.dot(answer_emb, context_emb) / (np.linalg.norm(answer_emb) * np.linalg.norm(context_emb) + 1e-8))

    if sim < config.MIN_GROUNDING_SCORE:
        return GuardrailResult(
            False, "output",
            f"Answer/context similarity {sim:.2f} is below the grounding floor "
            f"({config.MIN_GROUNDING_SCORE}); answer may not be grounded in retrieved context.",
            score=sim,
        )
    return GuardrailResult(True, "output", score=sim)
