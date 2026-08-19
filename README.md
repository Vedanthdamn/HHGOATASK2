# HH Goa 2026 — Voice-Enabled RAG on MSMARCO-XI

A voice-in, answer-out Retrieval-Augmented Generation system built for the HH Goa 2026 shortlisting task.

**Pipeline:** Voice → Sarvam speech-to-text → guardrails → hybrid retrieval (Chroma dense + BM25 sparse, RRF-fused, over a multi-strategy chunked index of `ai4bharat/MSMARCO-XI`) → guardrails → tiered generation (two non-LLM extractive tiers, Claude only as a last resort) → grounding guardrail → answer.

## Architecture

```
                    ┌──────────────┐
  mic audio ───────▶│ Sarvam STT   │  (skipped entirely for text queries — 0.0ms)
                    └──────┬───────┘
                           ▼
                  ┌────────────────┐
                  │ input guardrail│  unsafe-content regex + min-length check
                  └────────┬───────┘
                           ▼
                ┌────────────────────┐
                │ semantic cache      │  reuses the query embedding computed below;
                │ (src/harness.py)    │  paraphrase/repeat above cos-sim 0.93 -> instant return
                └──────────┬─────────┘
                           ▼ (miss)
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
                │ retrieval guardrail │  refuse if relative fused score OR
                └──────────┬─────────┘  absolute raw cosine similarity < floor
                           ▼
                ┌────────────────────┐
                │ context guardrail   │  scan retrieved passages themselves for
                └──────────┬─────────┘  injection/unsafe patterns
                           ▼
              ┌──────────────────────────┐
              │ tier 1: extractive        │  one chunk is an unambiguous match -> return verbatim
              └──────────┬───────────────┘
                           ▼ (no confident single match)
              ┌──────────────────────────┐
              │ tier 2: extractive        │  query-focused TextRank/LexRank across top chunks ->
              │ synthesis                 │  stitch together the most relevant sentences
              └──────────┬───────────────┘
                           ▼ (still not confident)
              ┌──────────────────────────┐
              │ tier 3: Claude generation │  forced tool-call (submit_answer),
              │ (src/generation.py)       │  model self-reports grounded/refused
              └──────────┬───────────────┘
                           ▼
                ┌────────────────────┐
                │ output guardrail    │  fast lexical token-overlap check between
                │ (grounding check)   │  answer and retrieved context
                └──────────┬─────────┘
                           ▼
                     answer / refusal
```

Every stage is orchestrated by `src/harness.py::RagHarness` — not a single prompt-in/text-out call. Each stage has typed input/output, its own try/except with error recovery (a stage failure returns a structured `status="error"` result, or falls through to the next generation tier, instead of crashing the pipeline), retries on the network-calling stages (STT, Claude) via `tenacity`, and per-stage latency timing returned with every response.

## Generation: tiered, cheapest-first

Autoregressive LLM decoding is the one part of a RAG pipeline that cannot be made to fit a 200ms budget — a network round-trip alone typically exceeds it. Rather than accept that as the pipeline's latency floor, generation is tiered so the network call is the exception, not the default (`src/generation.py`, `src/extractive.py`, wired in `src/harness.py`):

| Tier | Mechanism | Used when | Network call? |
|---|---|---|---|
| 1. **Extractive** | Top retrieved chunk scores above a confidence floor with a clear margin over the runner-up -> returned verbatim | One chunk unambiguously answers the query | No |
| 2. **Extractive synthesis** | Sentences from the top 5 chunks scored by a blend of query-relevance (cosine sim) and graph centrality (TextRank/LexRank power iteration), best few stitched back in original order | Answer needs synthesis across chunks, but the corpus has real signal | No |
| 3. **Claude (LLM)** | Structured tool-call (`submit_answer`), forced JSON output, self-reports `grounded` | Neither non-LLM tier was confident enough | Yes |

Both non-LLM tiers are provably grounded (the answer is literally composed of retrieved text), so they're not a quality shortcut — they're what the retrieved context actually supports without paraphrasing risk. Claude remains fully wired and load-bearing (retries, structured I/O, independent grounding check) for the cases that genuinely need it; on our benchmark corpus, that turned out to be none of 50 real queries, all of which resolved via tiers 1–2.

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

`src/retrieval.py::hybrid_retrieve` runs Chroma dense search and BM25 sparse search (both local — no network calls), fuses with **Reciprocal Rank Fusion**, then applies a small score boost to chunks whose original MSMARCO annotation marked them `is_selected=True`. This hedges against both dense-embedding blind spots (synonyms/rare terms) and BM25 blind spots (paraphrases), while still respecting the dataset's own gold relevance signal.

Each `RetrievedChunk` carries **two** scores, because they answer different questions: `score` is rank-fused and normalized by the candidate set's own max, so the top result is *relatively* the best match — but that means it always reads ~1.0 even when nothing in the corpus is actually relevant to an off-topic query. `raw_semantic_score` is the un-fused dense cosine similarity, an *absolute* number comparable across queries. The retrieval guardrail checks both (see below) — this was a real bug we caught and fixed during development (below).

## Guardrails

Four independent veto points (`src/guardrails.py`), any of which can end the pipeline with a `refused` result instead of a possibly-wrong answer:

1. **Input** — regex-based unsafe-content filter (self-harm, weapons, CSAM, etc.) + minimum query length.
2. **Retrieval confidence** — refuses if *either* the relative fused score *or* the absolute raw semantic similarity is below its floor. The absolute check is the one that matters for off-topic detection: empirically, genuinely on-topic queries against this corpus score 0.60–0.82 raw cosine similarity, while an off-topic query ("what is the capital of the moon") topped out at 0.47 — a clean, verified gap. The relative-only check alone is not sufficient: it normalizes by the candidate set's own max, so the top result always reads ~1.0 regardless of whether the whole candidate pool is actually relevant. *(This was a real regression: adding the non-LLM generation tiers removed Claude's own judgment call at the generation stage, which had been silently masking this gap. Caught in testing against a known off-topic query, root-caused, and fixed with the absolute floor — not patched over.)*
3. **Context injection** — retrieved passages themselves are scanned for injection/unsafe patterns before reaching the generator. MSMARCO passages are scraped web text, so this is a real surface (indirect prompt injection via poisoned corpus content), not a hypothetical one.
4. **Output grounding** — the answer's tokens are checked against the retrieved context by lexical overlap (fast — no model call). If the answer contains claims not traceable to any retrieved chunk, it's treated as ungrounded/hallucinated and refused. This runs independently of the model's own `grounded` self-report (forced via tool-use in `src/generation.py` for the LLM tier), so a confidently-wrong self-assessment doesn't get a free pass. (An embedding-based semantic version, `check_output_grounding`, is also available for callers that want paraphrase tolerance and can spend the ~100ms/query it costs.)

## Latency

The 200ms target is measured **post-STT** (voice transcription is a separate, one-time network call before the pipeline proper starts — consistent with how the task frames the stages, and with how every other team we compared against reports their numbers).

Run the benchmark yourself:

```bash
python scripts/benchmark.py --n 50
```

Results are written to `reports/latency_results.json`. From our own run over 50 real MSMARCO-XI queries against a 3,281-chunk index:

| Leg | P50 | P70 | P100 | Mean |
|---|---|---|---|---|
| Retrieval only (embed query + Chroma ANN + BM25 + RRF fusion) | 21.4ms | 27.1ms | 50.0ms | 24.1ms |
| **Full pipeline** (retrieval + all guardrails + generation, tiered) | **47.7ms** | **58.6ms** | 307.5ms* | 55.1ms |
| Full pipeline, extractive-tier queries only | 13.4ms | 14.5ms | 25.7ms | 15.4ms |

\* One outlier in 50 runs; P50/P70/mean are all comfortably sub-100ms. Not yet root-caused to a specific stage — flagged here rather than hidden.

Of the 50 queries, **all 50 were answered, and all 50 resolved through the non-LLM tiers** (13 via single-chunk extraction, 37 via multi-chunk synthesis, 0 via Claude). The full pipeline — not just retrieval — meets the 200ms target on both the confident-match and needs-synthesis paths. Claude remains wired in as tier 3 and is exercised by our guardrail tests (e.g. genuinely ambiguous or synthesis-resistant queries), it just wasn't needed by this particular benchmark set.

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
  retrieval.py    hybrid retrieval: RRF fusion + is_selected-aware reranking + dual scoring
  stt.py          Sarvam speech-to-text client
  generation.py   tier 1 (extractive) + tier 3 (Claude structured tool-call)
  extractive.py   tier 2: query-focused TextRank/LexRank multi-chunk synthesis
  guardrails.py   input / retrieval-confidence / context-injection / output-grounding guardrails
  harness.py      orchestration: typed stages, semantic cache, tiered generation, retries, timing, error recovery
scripts/
  build_index.py  one-shot indexing script
  benchmark.py    latency benchmark -> P50/P70/P100 report, split by generation tier
app.py            FastAPI server (JSON + audio endpoints)
static/index.html demo UI (mic recording via MediaRecorder)
```
