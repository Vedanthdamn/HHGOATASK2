"""Thin wrapper around a local sentence-transformers model.

Local (not API-based) so embedding never adds network latency to the
retrieval hot path, and so indexing doesn't burn API quota on ~thousands of
chunks. Model is multilingual so it works across the Indic-language configs
of MSMARCO-XI.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import config

_model_cache: dict[str, SentenceTransformer] = {}


def get_embedder(model_name: str = None) -> SentenceTransformer:
    model_name = model_name or config.EMBEDDING_MODEL
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class Embedder:
    def __init__(self, model_name: str = None):
        self.model = get_embedder(model_name)

    def encode(self, texts, normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def encode_one(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.encode([text], normalize=normalize)[0]
