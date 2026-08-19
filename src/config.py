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
    MIN_GROUNDING_SCORE = 0.28  # cosine sim floor between answer claim and context (embedding-based check)
    MIN_LEXICAL_GROUNDING_OVERLAP = 0.5  # fraction of answer tokens that must appear in context (fast default check)
    MIN_LEXICAL_DIVERSITY = 0.25  # unique words / total words; below this, treat answer text as degenerate/repetitive
    MIN_RETRIEVAL_SCORE = 0.20  # below this, treat as "no relevant context" (relative, rank-fused score)
    MIN_RAW_SEMANTIC_SCORE = 0.52  # below this, treat as off-topic (absolute, un-fused cosine similarity)

    # Semantic query cache: paraphrased/repeated queries above this cosine
    # similarity to a previously answered query skip retrieval+generation
    # entirely and return the cached answer.
    QUERY_CACHE_SIMILARITY_THRESHOLD = 0.93
    QUERY_CACHE_MAX_SIZE = 500

    # Extractive fast-path: skip the Claude network call entirely when the top
    # retrieved chunk is an unambiguous, high-confidence match. Only falls
    # back to LLM generation when the answer actually needs synthesis across
    # chunks or the match is uncertain -- this is what makes the *typical*
    # query's full pipeline (not just the retrieval leg) land under 200ms.
    EXTRACTIVE_CONFIDENCE_THRESHOLD = 0.85
    EXTRACTIVE_MARGIN = 0.15  # top score must also beat the runner-up by this much


config = Config()
