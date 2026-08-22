"""Thin wrapper around a local sentence-transformers model.

Local (not API-based) so embedding never adds network latency to the
retrieval hot path, and so indexing doesn't burn API quota on ~thousands of
chunks. Model is multilingual so it works across the Indic-language configs
of MSMARCO-XI.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import config
from src.onnx_encoder import OnnxEncoder

_model_cache: dict[str, SentenceTransformer] = {}


def get_embedder(model_name: str = None) -> SentenceTransformer:
    model_name = model_name or config.EMBEDDING_MODEL
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class Embedder:
    """Encodes text, preferring the ONNX graph and falling back to PyTorch.

    Both paths run the same weights and produce the same vectors (verified:
    cosine 1.00000000, max abs diff ~1e-7), so which one is active is purely
    a latency question -- ~8ms vs ~22ms per query single-threaded. The
    sentence-transformers model is loaded lazily so that when ONNX is
    available we don't pay to materialize a second copy of the weights.
    """

    def __init__(self, model_name: str = None, prefer_onnx: bool = True):
        self._model_name = model_name
        self._st_model = None
        self._onnx = OnnxEncoder.load_if_available() if prefer_onnx else None

    @property
    def model(self):
        """The sentence-transformers model, materialized on first use."""
        if self._st_model is None:
            self._st_model = get_embedder(self._model_name)
        return self._st_model

    @property
    def backend(self) -> str:
        return "onnx" if self._onnx is not None else "torch"

    def encode(self, texts, normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self._onnx is not None:
            try:
                return self._onnx.encode(list(texts), normalize=normalize)
            except Exception:  # noqa: BLE001 - fall back rather than fail a request
                self._onnx = None
        return self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def encode_one(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.encode([text], normalize=normalize)[0]
