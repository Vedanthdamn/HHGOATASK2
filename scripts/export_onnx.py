"""Export the query encoder to ONNX for the low-latency inference path.

Run at image build time (see Dockerfile): the graph carries the model's
weights (~449MB, dominated by the XLM-R vocabulary embedding), which is far
too large to keep in git. The runtime loader is best-effort, so if this
step is skipped the app still works -- it just falls back to PyTorch and is
~2.7x slower to embed a query.

Usage:
    python scripts/export_onnx.py [--out /app/onnx_model]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import config
from src.embeddings import get_embedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/app/onnx_model")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    st_model = get_embedder()
    transformer = st_model[0].auto_model
    transformer.eval()
    tokenizer = st_model.tokenizer
    max_seq_length = int(getattr(st_model, "max_seq_length", 128) or 128)

    sample = tokenizer("नमूना प्रश्न", return_tensors="pt", padding=True, truncation=True)
    model_path = out_dir / "query_encoder.onnx"

    print(f"Exporting {config.EMBEDDING_MODEL} -> {model_path}")
    torch.onnx.export(
        transformer,
        (sample["input_ids"], sample["attention_mask"]),
        str(model_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        # Batch and sequence length both vary: single short queries at request
        # time, larger batches when building the sentence index.
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # The tokenizer travels with the graph so the runtime never needs the
    # sentence-transformers model directory or a network call to load.
    tokenizer_dir = out_dir / "tokenizer"
    if tokenizer_dir.exists():
        shutil.rmtree(tokenizer_dir)
    tokenizer.save_pretrained(str(tokenizer_dir))

    (out_dir / "query_encoder.json").write_text(json.dumps({
        "model_name": config.EMBEDDING_MODEL,
        "max_seq_length": max_seq_length,
        "tokenizer_dir": str(tokenizer_dir),
        "pooling": "mean",
    }, indent=2))

    total_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
    print(f"Exported to {out_dir} ({total_mb:.0f} MB), max_seq_length={max_seq_length}")


if __name__ == "__main__":
    main()
