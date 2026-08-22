"""ONNX Runtime backend for the query encoder.

After the retrieval and synthesis fixes, ~95% of end-to-end latency was a
single transformer forward pass to embed the query. That pass is small
(one short sentence through a 12-layer MiniLM), so most of its cost is
PyTorch dispatch overhead rather than arithmetic -- exactly what an
ahead-of-time compiled graph removes.

Running the *same weights* through ONNX Runtime instead measured 21.7ms ->
8.0ms single-threaded (2.7x), and the vectors are the same ones: cosine
1.00000000 against the PyTorch output, max absolute difference ~1e-7. This
is a runtime change, not a model change, so nothing about retrieval quality
moves and the existing index stays valid.

The exported graph is ~449MB (the XLM-R vocabulary embedding dominates), so
it is built at image build time by `scripts/export_onnx.py` rather than
committed. Everything here is best-effort: if the artifact or onnxruntime
is missing, `load_if_available` returns None and `Embedder` transparently
falls back to sentence-transformers.
"""
import json
import os

import numpy as np

DEFAULT_DIR = os.environ.get("ONNX_ENCODER_DIR", "/app/onnx_model")
MODEL_FILE = "query_encoder.onnx"
META_FILE = "query_encoder.json"


class OnnxEncoder:
    def __init__(self, session, tokenizer, max_seq_length: int):
        self._session = session
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length

    @classmethod
    def load_if_available(cls, directory: str = None) -> "OnnxEncoder | None":
        directory = directory or DEFAULT_DIR
        model_path = os.path.join(directory, MODEL_FILE)
        meta_path = os.path.join(directory, META_FILE)
        if not (os.path.exists(model_path) and os.path.exists(meta_path)):
            return None
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            with open(meta_path) as f:
                meta = json.load(f)

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Match the container's thread budget; oversubscribing a 2-vCPU box
            # costs more in contention than it wins in parallelism.
            threads = int(os.environ.get("ONNX_INTRA_THREADS", os.environ.get("OMP_NUM_THREADS", "1")))
            options.intra_op_num_threads = threads
            session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
            tokenizer = AutoTokenizer.from_pretrained(meta["tokenizer_dir"])
            return cls(session, tokenizer, int(meta["max_seq_length"]))
        except Exception:  # noqa: BLE001 - an unusable accelerator must never break startup
            return None

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        encoded = self._tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self._max_seq_length,  # mirror sentence-transformers' own cap
        )
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        hidden = self._session.run(
            ["last_hidden_state"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]

        # Mean pooling over non-padding tokens -- the pooling mode this model's
        # sentence-transformers config declares.
        mask = attention_mask[..., None].astype(np.float32)
        vectors = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        if normalize:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors.astype(np.float32)
