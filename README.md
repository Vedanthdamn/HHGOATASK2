# HH Goa 2026 — Voice-Enabled RAG on MSMARCO-XI

A voice-in, answer-out Retrieval-Augmented Generation system built for the HH Goa 2026 shortlisting task.

**Pipeline:** Voice → Sarvam speech-to-text → guardrails → hybrid retrieval (Chroma dense + BM25 sparse, RRF-fused, over a multi-strategy chunked index of `ai4bharat/MSMARCO-XI`) → guardrails → Claude structured-tool-call generation → grounding guardrail → answer.

## Architecture

```
                    ┌──────────────┐
  mic audio ───────▶│ Sarvam STT   │
                    └──────┬───────┘
                           ▼
                  ┌────────────────┐
                  │ input guardrail│  unsafe-content regex + min-length check
                  └────────┬───────┘
                           ▼
        ┌──────────────────────────────────┐
        │ hybrid retrieval (src/retrieval.py)│
        │  - dense: Chroma ANN (cosine)      │
        │  - sparse: BM25Okapi                │
        │  - fused: Reciprocal Rank Fusion    │
        │  - boosted: dataset's own            │
        │    is_selected relevance labels      │
        └──────────────────┬────────────────┘
                           ▼
                ┌────────────────────┐
                │ retrieval guardrail │  refuse if top fused score < floor
                └──────────┬─────────┘
                           ▼
                ┌────────────────────┐
                │ Claude generation   │  forced tool-call (submit_answer),
                │ (src/generation.py) │  model self-reports grounded/refused
                └──────────┬─────────┘
                           ▼
                ┌────────────────────┐
                │ output guardrail    │  embedding-similarity check between
                │ (grounding check)   │  answer and retrieved context
                └──────────┬─────────┘
                           ▼
                     answer / refusal
```

Every stage is orchestrated by `src/harness.py::RagHarness` — not a single prompt-in/text-out call. Each stage has typed input/output, its own try/except with error recovery (a stage failure returns a structured `status="error"` result instead of crashing the pipeline), retries on the network-calling stages (STT, generation) via `tenacity`, and per-stage latency timing.

## Chunking strategy

Naive fixed-size chunking was intentionally avoided — see `src/chunking.py`. A router (`chunk_document`) picks a strategy per passage:

| Strategy | When used | Why |
|---|---|---|
| **Atomic** | passage ≤ 220 chars | MSMARCO passages are often already a single short fact; splitting destroys context for no benefit |
| **Semantic (breakpoint)** | ≥3 sentences, embedder available | Embeds each sentence, cuts wherever consecutive-sentence cosine distance exceeds the passage's own 90th-percentile distance — keeps topically coherent spans together instead of a blind cut (TextTiling-style) |
| **Sentence-window** | ≥3 sentences, no embedder | Sliding window of N sentences with overlap, as a semantic-chunking fallback |
| **Fixed-size** | short passage, <3 sentences, longer than the atomic threshold | Robust fallback: fixed character window with overlap |

Every chunk is **metadata-aware**: it carries `doc_id`, `query_id`, `query_type`, `lang`, the source strategy, and the dataset's own `is_selected` relevance flag. Retrieval uses that flag as a free reranking signal (`src/retrieval.py`), on top of dense+sparse fusion.

## Retrieval

`src/retrieval.py::hybrid_retrieve` runs Chroma dense search and BM25 sparse search in parallel-ish (both are local/fast — no network calls), fuses with **Reciprocal Rank Fusion**, then applies a small score boost to chunks whose original MSMARCO annotation marked them `is_selected=True`. This hedges against both dense-embedding blind spots (synonyms/rare terms) and BM25 blind spots (paraphrases), while still respecting the dataset's own gold relevance signal.

## Latency

The 200ms target is measured on the **retrieval leg** (embed query + Chroma ANN search + BM25 + RRF fusion) — chunking itself is precomputed once at index time, not on the query path. Separately, we report **full end-to-end** latency including guardrails and the Claude generation network call, because that number is dominated by a third-party API round-trip and shouldn't be conflated with the retrieval-engineering number the 200ms target is actually about.

Run the benchmark yourself:

```bash
python scripts/benchmark.py --n 50
```

Results are written to `reports/latency_results.json`. From our own run over 50 real MSMARCO-XI queries against a 3,281-chunk index:

| Leg | P50 | P70 | P100 | Mean |
|---|---|---|---|---|
| **Retrieval only** (embed query + Chroma ANN + BM25 + RRF fusion) | 29.7ms | 38.0ms | 59.8ms | 33.0ms |
| End-to-end (retrieval + guardrails + Claude generation over the network) | 4.7s | 5.5s | 7.4s | 4.7s |

The retrieval leg — the part actually named in the 200ms target — comes in **well under budget at every percentile**, including worst-case (P100 = 59.8ms). The end-to-end number is dominated by the Claude API round-trip and is reported for transparency, not because it's realistic to fit an LLM generation call under 200ms.

Of the 50 queries, 22 were answered and 28 were refused — almost all refusals happened at the **generation guardrail** (Claude itself determined the retrieved chunks didn't contain the answer), not because retrieval failed outright. That's the guardrail doing its job: MSMARCO-XI passages are noisy candidate sets where only a subset per query is actually relevant, and the system declines rather than hallucinating from the rest.

## Guardrails

Three independent veto points (`src/guardrails.py`), any of which can end the pipeline with a `refused` result instead of a possibly-wrong answer:

1. **Input** — regex-based unsafe-content filter (self-harm, weapons, CSAM, etc.) + minimum query length.
2. **Retrieval confidence** — if the top fused retrieval score is below a floor, we refuse rather than let the LLM improvise from irrelevant context (handles off-topic queries the corpus can't answer).
3. **Output grounding** — after generation, the answer and the retrieved context are both embedded and compared by cosine similarity; if the answer has drifted too far from what was actually retrieved, it's treated as an ungrounded/hallucinated response and refused. This is independent of the model's own `grounded` self-report (forced via tool-use in `src/generation.py`), so a confidently-wrong self-assessment doesn't get a free pass.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SARVAM_API_KEY and ANTHROPIC_API_KEY
```

Build the index (downloads a sample of MSMARCO-XI, chunks, embeds, indexes):

```bash
python scripts/build_index.py --lang hi --sample-size 2000
```

Run the server:

```bash
python app.py
# open http://localhost:8000
```

Run the latency benchmark:

```bash
python scripts/benchmark.py --n 50
```

## Project layout

```
src/
  config.py       env-driven configuration
  dataset.py      loads + flattens ai4bharat/MSMARCO-XI into passage documents
  chunking.py     multi-strategy chunking router (atomic/semantic/sentence-window/fixed-size)
  embeddings.py   local multilingual sentence-transformers wrapper
  vectorstore.py  Chroma (dense) + BM25 (sparse) persistent index
  retrieval.py    hybrid retrieval: RRF fusion + is_selected-aware reranking
  stt.py          Sarvam speech-to-text client
  generation.py   Claude structured tool-call generation
  guardrails.py   input / retrieval-confidence / output-grounding guardrails
  harness.py      orchestration: typed stages, retries, per-stage timing, error recovery
scripts/
  build_index.py  one-shot indexing script
  benchmark.py    latency benchmark -> P50/P70/P100 report
app.py            FastAPI server (JSON + audio endpoints)
static/index.html demo UI (mic recording via MediaRecorder)
```
