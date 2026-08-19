import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    DATASET_NAME = os.getenv("DATASET_NAME", "ai4bharat/MSMARCO-XI")
    DATASET_LANG = os.getenv("DATASET_LANG", "hi")
    DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train")
    DATASET_SAMPLE_SIZE = int(os.getenv("DATASET_SAMPLE_SIZE", "2000"))

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")

    # Retrieval tuning
    TOP_K_SEMANTIC = 8
    TOP_K_BM25 = 8
    TOP_K_FINAL = 5
    RRF_K = 60  # reciprocal rank fusion constant

    # Guardrails
    MIN_GROUNDING_SCORE = 0.28  # cosine sim floor between answer claim and context
    MIN_RETRIEVAL_SCORE = 0.20  # below this, treat as "no relevant context"


config = Config()
