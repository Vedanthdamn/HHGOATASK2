"""FastAPI app: voice-enabled RAG demo.

POST /ask        - JSON {"query": "..."} text-only path (useful for testing/benchmarking)
POST /ask-audio   - multipart audio file -> Sarvam STT -> full pipeline
GET  /health      - liveness + index size
GET  /            - static demo UI (mic recording -> pipeline -> answer)
"""
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.embeddings import Embedder
from src.harness import RagHarness
from src.vectorstore import VectorStore

app = FastAPI(title="HH Goa Voice RAG")

_embedder = Embedder()
_store = VectorStore(embedder=_embedder)
_harness = RagHarness(store=_store, embedder=_embedder)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AskRequest(BaseModel):
    query: str


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": _store.count()}


@app.post("/ask")
def ask(req: AskRequest):
    result = _harness.run(query=req.query)
    return JSONResponse(result.to_dict())


@app.post("/ask-audio")
async def ask_audio(file: UploadFile = File(...), language_code: str = Form("hi-IN")):
    audio_bytes = await file.read()
    result = _harness.run(audio_bytes=audio_bytes, language_code=language_code)
    return JSONResponse(result.to_dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
