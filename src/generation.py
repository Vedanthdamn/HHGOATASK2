"""Answer generation via the Claude API, forced into structured JSON output
via tool-use (rather than a raw prompt-in/text-out call), so the harness can
reason about citations and the model's own grounded/refused self-assessment
without parsing free text.
"""
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import config
from src.retrieval import RetrievedChunk

_client = None

_SUBMIT_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit the final answer to the user's question, grounded strictly in the provided context passages.",
    "input_schema": {
        "type": "object",
        "properties": {
            "grounded": {
                "type": "boolean",
                "description": "True only if the answer is fully supported by the provided context passages.",
            },
            "answer": {
                "type": "string",
                "description": "The answer to the user's question, or an empty string if not grounded/answerable.",
            },
            "cited_chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "chunk_id values of the context passages actually used to construct the answer.",
            },
            "refusal_reason": {
                "type": ["string", "null"],
                "description": "If grounded is false, a short reason why the question can't be answered from context.",
            },
        },
        "required": ["grounded", "answer", "cited_chunk_ids"],
    },
}

SYSTEM_PROMPT = (
    "You are a retrieval-augmented question answering assistant. "
    "You will be given a user question and a set of retrieved context passages, each with a chunk_id. "
    "Answer ONLY using information present in the context passages. "
    "Do not use outside knowledge, do not speculate, and do not fill gaps with assumptions. "
    "If the context does not contain enough information to answer confidently, set grounded=false, "
    "leave answer empty, and give a refusal_reason. "
    "Always respond by calling the submit_answer tool."
)


def try_extractive_answer(chunks: list[RetrievedChunk]) -> dict | None:
    """Non-LLM fast path: if the top chunk is an unambiguous, high-confidence
    match (score above a floor AND a clear margin over the runner-up), return
    it directly as the answer instead of paying for an LLM decode.

    This is what makes the *typical* query's full pipeline -- not just the
    retrieval leg -- land under 200ms: no network call at all for the
    confident case. Ambiguous or multi-chunk-synthesis cases still fall
    through to `generate_answer` (the LLM path), so nothing here trades away
    quality on hard questions -- it only short-circuits the easy ones.
    """
    if not chunks:
        return None

    top = chunks[0]
    if top.score < config.EXTRACTIVE_CONFIDENCE_THRESHOLD:
        return None

    runner_up_score = chunks[1].score if len(chunks) > 1 else 0.0
    if (top.score - runner_up_score) < config.EXTRACTIVE_MARGIN:
        return None

    return {
        "grounded": True,
        "answer": top.text,
        "cited_chunk_ids": [top.chunk_id],
        "refusal_reason": None,
        "mode": "extractive",
    }


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for c in chunks:
        lines.append(f"[chunk_id={c.chunk_id}] {c.text}")
    return "\n".join(lines)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.3, min=0.3, max=2))
def generate_answer(query: str, chunks: list[RetrievedChunk]) -> dict:
    context_block = _build_context_block(chunks)
    user_msg = f"Question: {query}\n\nContext passages:\n{context_block}"

    client = get_client()
    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        tools=[_SUBMIT_ANSWER_TOOL],
        tool_choice={"type": "tool", "name": "submit_answer"},
        messages=[{"role": "user", "content": user_msg}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_answer":
            result = dict(block.input)
            result.setdefault("cited_chunk_ids", [])
            result.setdefault("refusal_reason", None)
            result["mode"] = "llm"
            return result

    raise RuntimeError("Claude did not return a submit_answer tool call")
