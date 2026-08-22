FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY static/ static/
COPY scripts/ scripts/
COPY app.py .
COPY data/ data/
COPY reports/ reports/

# Export the query encoder to ONNX at build time. The graph embeds the model
# weights (~449MB, mostly the XLM-R vocabulary matrix), which is far too large
# for git, so it is produced here instead of committed. Embedding the query is
# ~95% of end-to-end latency and ONNX Runtime does it ~2.7x faster than
# PyTorch with byte-identical vectors -- see src/onnx_encoder.py.
# Best-effort: if the export fails the app still runs, just on the slower
# PyTorch path, so a broken export must not break the image.
ENV ONNX_ENCODER_DIR=/app/onnx_model
RUN python scripts/export_onnx.py --out /app/onnx_model || \
    echo "ONNX export failed; falling back to the PyTorch encoder at runtime"

# Stamped by CI so /health can report exactly which build is live -- a failed
# image pull once left a stale container running and reporting healthy.
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

ENV CHROMA_PERSIST_DIR=/app/data/chroma
ENV PORT=7860
# Cap thread pools: default OMP/MKL behavior spawns one thread per core and
# allocates per-thread buffers, which adds up fast on memory-constrained
# hosts (Render free tier caps at 512MB) without materially helping latency
# at our request volume.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
EXPOSE 7860

CMD ["python", "app.py"]
