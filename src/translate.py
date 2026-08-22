"""Answer translation via Sarvam AI's /translate endpoint.

This is a display-side feature, not part of the RAG pipeline: the corpus,
retrieval, and generation are all Hindi (MSMARCO-XI[hi]), and this only
translates the already-generated answer for the demo UI's language dropdown.
It runs after the 200ms-budgeted pipeline has already returned, so it isn't
subject to that target.
"""
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import config

SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"

# UI-facing dropdown options -> Sarvam's BCP-47-ish codes.
SUPPORTED_LANGUAGES = {
    "hi-IN": "Hindi (original)",
    "en-IN": "English",
    "te-IN": "Telugu",
    "ta-IN": "Tamil",
}


class TranslateError(Exception):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.3, min=0.3, max=2))
def translate_text(text: str, target_language_code: str, source_language_code: str = "auto") -> dict:
    if not config.SARVAM_API_KEY:
        raise TranslateError("SARVAM_API_KEY is not set")
    if target_language_code not in SUPPORTED_LANGUAGES:
        raise TranslateError(f"Unsupported target language: {target_language_code}")
    if not text or not text.strip():
        raise TranslateError("No text to translate")

    payload = {
        "input": text[:2000],  # Sarvam's documented input cap
        "source_language_code": source_language_code,
        "target_language_code": target_language_code,
        "model": "mayura:v1",
    }
    headers = {"Content-Type": "application/json", "api-subscription-key": config.SARVAM_API_KEY}

    resp = requests.post(SARVAM_TRANSLATE_URL, json=payload, headers=headers, timeout=15)
    if resp.status_code != 200:
        raise TranslateError(f"Sarvam translate failed ({resp.status_code}): {resp.text[:300]}")

    body = resp.json()
    translated = (body.get("translated_text") or "").strip()
    if not translated:
        raise TranslateError("Sarvam translate returned empty output")

    return {"translated_text": translated, "source_language_code": body.get("source_language_code", source_language_code)}
