"""FastAPI app: voice-enabled RAG demo.

POST /ask        - JSON {"query": "..."} text-only path (useful for testing/benchmarking)
POST /ask-audio   - multipart audio file -> Sarvam STT -> full pipeline
POST /translate   - JSON {"text": "...", "target_language_code": "en-IN"} -> translated answer
                    (display-side only: the pipeline itself always runs in Hindi)
GET  /health      - liveness + index size
GET  /metrics     - P50/P70/P100 latency benchmark results (scripts/benchmark.py output)
GET  /            - landing/overview page with a Task #2 card -> /app
GET  /app         - the actual demo UI (mic recording -> pipeline -> answer)
"""
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.embeddings import Embedder
from src.harness import RagHarness
from src.translate import SUPPORTED_LANGUAGES, TranslateError, translate_text
from src.vectorstore import VectorStore

app = FastAPI(title="HH Goa Voice RAG")

_embedder = Embedder()
_store = VectorStore(embedder=_embedder)
_harness = RagHarness(store=_store, embedder=_embedder)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

REPORTS_DIR = Path(__file__).parent / "reports"
LATENCY_REPORT_PATH = REPORTS_DIR / "latency_results.json"


class AskRequest(BaseModel):
    query: str


class TranslateRequest(BaseModel):
    text: str
    target_language_code: str


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/app")
def app_ui():
    return FileResponse(str(STATIC_DIR / "app.html"))


@app.get("/health")
def health():
    # `fast_index` reports whether the in-memory exact search index built. It
    # falls back to the Chroma/rank_bm25 path silently when it can't (correct,
    # but ~15x slower), so surface it rather than let a silent fallback look
    # like a healthy deploy.
    return {
        "status": "ok",
        "version": os.environ.get("GIT_SHA", "unknown"),
        "indexed_chunks": _store.count(),
        "fast_index": len(_store.fast) if _store.fast is not None else None,
        "sentence_index": len(_harness.sentence_index),
        "encoder_backend": _embedder.backend,  # "onnx" (fast path) or "torch" (fallback)
    }


@app.get("/metrics")
def metrics():
    if not LATENCY_REPORT_PATH.exists():
        return JSONResponse({"error": "No benchmark report found. Run scripts/benchmark.py first."}, status_code=404)
    return JSONResponse(json.loads(LATENCY_REPORT_PATH.read_text()))


@app.post("/ask")
def ask(req: AskRequest):
    result = _harness.run(query=req.query)
    return JSONResponse(result.to_dict())


@app.post("/ask-audio")
async def ask_audio(file: UploadFile = File(...), language_code: str = Form("hi-IN")):
    audio_bytes = await file.read()
    result = _harness.run(audio_bytes=audio_bytes, language_code=language_code, audio_filename=file.filename or "audio.wav")
    return JSONResponse(result.to_dict())


@app.post("/translate")
def translate(req: TranslateRequest):
    if req.target_language_code not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported target_language_code. Choose from: {list(SUPPORTED_LANGUAGES)}")
    try:
        result = translate_text(req.text, req.target_language_code)
    except TranslateError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
