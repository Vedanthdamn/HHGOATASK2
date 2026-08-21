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

ENV CHROMA_PERSIST_DIR=/app/data/chroma
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
