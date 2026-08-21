FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY static/ static/
COPY app.py .
COPY data/ data/
COPY reports/ reports/

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
