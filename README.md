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

Five independent veto points (`src/guardrails.py`), any of which can end the pipeline with a `refused` result instead of a possibly-wrong answer:

1. **Input** — regex-based unsafe-content filter (self-harm, weapons, CSAM, etc.) + minimum query length.
2. **Retrieval confidence** — refuses if *either* the relative fused score *or* the absolute raw semantic similarity is below its floor. The absolute check is the one that matters for off-topic detection: empirically, genuinely on-topic queries against this corpus score 0.60–0.82 raw cosine similarity, while an off-topic query ("what is the capital of the moon") topped out at 0.47 — a clean, verified gap. The relative-only check alone is not sufficient: it normalizes by the candidate set's own max, so the top result always reads ~1.0 regardless of whether the whole candidate pool is actually relevant. *(This was a real regression: adding the non-LLM generation tiers removed Claude's own judgment call at the generation stage, which had been silently masking this gap. Caught in testing against a known off-topic query, root-caused, and fixed with the absolute floor — not patched over.)*
3. **Context injection** — retrieved passages themselves are scanned for injection/unsafe patterns before reaching the generator. MSMARCO passages are scraped web text, so this is a real surface (indirect prompt injection via poisoned corpus content), not a hypothetical one.
4. **Output grounding** — the answer's tokens are checked against the retrieved context by lexical overlap (fast — no model call). If the answer contains claims not traceable to any retrieved chunk, it's treated as ungrounded/hallucinated and refused. This runs independently of the model's own `grounded` self-report (forced via tool-use in `src/generation.py` for the LLM tier), so a confidently-wrong self-assessment doesn't get a free pass. (An embedding-based semantic version, `check_output_grounding`, is also available for callers that want paraphrase tolerance and can spend the ~100ms/query it costs.)
5. **Output coherence** — catches degenerate/repetitive text (lexical diversity floor: unique words / total words). Found live: a chunk in the translated corpus contains a neural-MT decoding-loop artifact — the same phrase repeated hundreds of times — that passed the grounding check trivially (an extractive answer is, by construction, grounded in itself) despite being unreadable garbage. This guardrail refuses before it reaches the user instead of returning it as a confident answer.

## Latency

The 200ms target is measured **post-STT** (voice transcription is a separate, one-time network call before the pipeline proper starts — consistent with how the task frames the stages, and with how every other team we compared against reports their numbers).

Run the benchmark yourself:

```bash
python scripts/benchmark.py --n 50
```

Results are written to `reports/latency_results.json`. Measured over 50 real MSMARCO-XI queries against a 3,281-chunk index, **inside the same container image that is deployed** (single-threaded BLAS, 2 vCPU) rather than on a developer laptop — so these are production numbers, not best-case local ones:

| Leg | P50 | P70 | **P100** | Mean |
|---|---|---|---|---|
| Retrieval only (embed query + Chroma ANN + BM25 + RRF fusion) | 39.5ms | 42.1ms | 66.5ms | 41.0ms |
| **Full pipeline** (retrieval + all guardrails + generation, tiered) | **41.6ms** | **43.8ms** | **62.8ms** | 43.4ms |
| Full pipeline, single-chunk extractive tier only | 42.0ms | 47.5ms | 53.3ms | 43.7ms |
| Full pipeline, multi-chunk synthesis tier only | 41.6ms | 43.8ms | 62.8ms | 43.3ms |

**Every percentile — including P100 — is inside the 200ms budget**, with ~3x headroom at the worst case. All 50 queries resolved through the non-LLM tiers (9 via single-chunk extraction, 41 via multi-chunk synthesis, 0 via Claude); 46 answered, 4 correctly refused by the retrieval-confidence guardrail. Claude remains wired in as tier 3 and is exercised by our guardrail tests, it just wasn't needed by this benchmark set.

### How the P100 was fixed (it used to be 307ms)

An earlier revision of this README reported a 307ms P100 outlier as "not yet root-caused." It is now root-caused and fixed, and the fix is the most interesting piece of engineering in the repo.

Profiling per stage showed the cost was not retrieval, not the LLM, and not the degenerate-chunk case we had assumed — it was **tier 2 embedding sentences on the query path**. `try_extractive_synthesis` scores individual sentences from the top 5 chunks, and it was calling the sentence-transformer on all of them per request: a *second* forward pass, measured at **613–864ms**, i.e. ~95% of end-to-end latency on the path 41 of 50 queries take. (Because the old benchmark bucketed only `extractive` and `llm` timings, the synthesis tier — the dominant path — had no bucket of its own and was invisible in the report. It now has one.)

The corpus is static, so that work does not belong on the hot path at all. `src/sentence_index.py` splits every indexed chunk into sentences **once at index time**, embeds them, and persists the vectors alongside the Chroma index (7,384 sentences over 3,270 chunks, 7.7MB). At query time tier 2 looks up the stored vectors for the already-retrieved chunks and reuses the query embedding computed upstream for retrieval — so it does **zero model calls**, just two numpy products. That tier went from ~600ms to ~0.4ms.

Because the vectors are computed with the same model and the same normalization, this is a pure latency change: we verified **all 50 answers are byte-identical** between the precomputed and on-the-fly paths (status, generation tier, and answer text), so it is not a quality/speed trade.

One further bug surfaced while validating it, and is worth recording because it is the kind that hides easily: the first version of the lookup fell back to on-the-fly embedding **all-or-nothing**, so a single chunk missing from the sentence index (11 chunks split into zero usable sentences and were skipped at build time) discarded every precomputed vector for that query and re-embedded all five chunks — 193ms instead of 0.4ms, reproducibly, leaving a 232ms P100. The fallback is now **per chunk**: an unindexed chunk costs only its own sentences, and a stale or absent index degrades latency gracefully instead of falling off a cliff. That is what took P100 from 232ms to 63ms.

```bash
python scripts/build_sentence_index.py   # run after build_index.py, or whenever the chunk index is rebuilt
```

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
python scripts/build_sentence_index.py   # precomputes tier-2 sentence vectors; see Latency
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
  sentence_index.py  sentence-level embeddings precomputed at index time, so tier 2 makes no model call
  retrieval.py    hybrid retrieval: RRF fusion + is_selected-aware reranking + dual scoring
  stt.py          Sarvam speech-to-text client
  generation.py   tier 1 (extractive) + tier 3 (Claude structured tool-call)
  extractive.py   tier 2: query-focused TextRank/LexRank multi-chunk synthesis
  guardrails.py   input / retrieval-confidence / context-injection / output-grounding / output-coherence guardrails
  harness.py      orchestration: typed stages, semantic cache, tiered generation, retries, timing, error recovery
scripts/
  build_index.py  one-shot indexing script
  build_sentence_index.py  precomputes tier-2 sentence embeddings off the query path
  benchmark.py    latency benchmark -> P50/P70/P100 report, split by generation tier
app.py            FastAPI server (JSON + audio endpoints)
static/index.html demo UI (mic recording via MediaRecorder)
```
