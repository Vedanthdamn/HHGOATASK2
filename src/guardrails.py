"""Guardrails: input safety/relevance checks, retrieval-confidence gating,
and post-hoc grounding verification on the generated answer.

Three independent layers, each of which can veto the pipeline before it
returns an answer to the user:

  1. INPUT   - blocks unsafe content and obviously off-topic chit-chat before
               we spend a retrieval + generation call on it.
  2. RETRIEVAL - if nothing in the corpus is actually relevant (low fused
               score), refuse rather than let the LLM improvise an answer.
  3. OUTPUT  - after generation, embed the answer and compare it against the
               retrieved context; if the answer drifts too far from what was
               actually retrieved, treat it as an ungrounded/hallucinated
               response and refuse instead of returning it.
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
    return GuardrailResult(True, "retrieval", score=top_score)


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
