"""Speech-to-text via Sarvam AI's /speech-to-text endpoint."""
import io

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import config

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class STTError(Exception):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.3, min=0.3, max=2))
def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav", language_code: str = "hi-IN") -> dict:
    if not config.SARVAM_API_KEY:
        raise STTError("SARVAM_API_KEY is not set")

    files = {"file": (filename, io.BytesIO(audio_bytes), "audio/wav")}
    data = {"model": "saarika:v2.5", "language_code": language_code}
    headers = {"api-subscription-key": config.SARVAM_API_KEY}

    resp = requests.post(SARVAM_STT_URL, headers=headers, files=files, data=data, timeout=15)
    if resp.status_code != 200:
        raise STTError(f"Sarvam STT failed ({resp.status_code}): {resp.text[:300]}")

    payload = resp.json()
    transcript = payload.get("transcript", "").strip()
    if not transcript:
        raise STTError("Sarvam STT returned an empty transcript")

    return {"transcript": transcript, "language_code": payload.get("language_code", language_code), "raw": payload}
