"""Loads and flattens the ai4bharat/MSMARCO-XI dataset into retrievable documents.

The dataset is published as one parquet file per language per split
(e.g. `train/hintrain.parquet`), not as separate HuggingFace `datasets`
configs, and its `passages` column is a nested struct
(`{English_passages: [...], Translated_passages: [...], is_selected: [...]}`)
that a couple of pyarrow/pandas parquet readers choke on
("Nested data conversions not implemented for chunked array outputs").
DuckDB reads it cleanly, so we fetch the single needed parquet file via
`huggingface_hub.hf_hub_download` (cached locally after the first run) and
query it directly with DuckDB rather than going through `datasets.load_dataset`.

Each row is a (query, passages) pair; `is_selected` flags which candidate
passages actually answer the query. We flatten every passage into its own
document, carrying query/answer/selection metadata along for evaluation and
for metadata-aware chunking downstream.
"""
from dataclasses import dataclass, field

import duckdb
from huggingface_hub import hf_hub_download

from src.config import config

# 2-letter code the rest of the app uses -> 3-letter filename prefix used in the repo
LANG_FILE_PREFIX = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}
SPLIT_SUFFIX = {"train": "train", "validation": "val"}


@dataclass
class Document:
    doc_id: str
    text: str
    query: str
    answer: str
    is_selected: bool
    query_id: str
    query_type: str
    lang: str
    metadata: dict = field(default_factory=dict)


def _resolve_parquet_path(lang: str, split: str) -> str:
    prefix = LANG_FILE_PREFIX.get(lang, lang)  # allow passing the 3-letter code directly too
    suffix = SPLIT_SUFFIX.get(split, split)
    relpath = f"{split}/{prefix}{suffix}.parquet"
    return hf_hub_download(config.DATASET_NAME, relpath, repo_type="dataset")


def load_msmarco_documents(
    lang: str = None,
    split: str = None,
    sample_size: int = None,
) -> list[Document]:
    lang = lang or config.DATASET_LANG
    split = split or config.DATASET_SPLIT
    sample_size = sample_size or config.DATASET_SAMPLE_SIZE

    path = _resolve_parquet_path(lang, split)

    # Fetch a generous row multiple up front (rows -> multiple passages each)
    # rather than loading the full multi-hundred-thousand-row file into memory.
    row_limit = max(sample_size, 200)
    con = duckdb.connect()
    cols = [d[0] for d in con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 0").description]

    documents: list[Document] = []
    rows_seen = 0

    while len(documents) < sample_size:
        batch_sql = f"SELECT * FROM read_parquet('{path}') LIMIT {row_limit} OFFSET {rows_seen}"
        rows = con.execute(batch_sql).fetchall()
        if not rows:
            break

        for row in rows:
            rows_seen += 1
            record = dict(zip(cols, row))
            query = (record.get("query") or "").strip()
            answer = (record.get("Answer") or "").strip()
            query_id = str(record.get("query_id", rows_seen))
            query_type = record.get("query_type", "unknown")

            passages_struct = record.get("passages") or {}
            passages = passages_struct.get("Translated_passages") or []
            selected_flags = passages_struct.get("is_selected") or []

            for i, passage in enumerate(passages):
                if not passage or not passage.strip():
                    continue
                is_selected = bool(selected_flags[i]) if i < len(selected_flags) else False
                documents.append(
                    Document(
                        doc_id=f"{query_id}-p{i}",
                        text=passage.strip(),
                        query=query,
                        answer=answer,
                        is_selected=is_selected,
                        query_id=query_id,
                        query_type=str(query_type),
                        lang=lang,
                        metadata={"passage_index": i, "source_row": rows_seen},
                    )
                )
                if len(documents) >= sample_size:
                    break
            if len(documents) >= sample_size:
                break

    return documents[:sample_size]


def load_msmarco_queries(lang: str = None, split: str = None, n: int = 50) -> list[str]:
    """Fetch the first N queries in file order (one row = one query).

    Deliberately NOT `SELECT DISTINCT` (which reorders via hashing): this
    must return the same leading rows that `load_msmarco_documents` indexes,
    so benchmark queries actually have matching passages in the index
    instead of legitimately-but-uninterestingly triggering the
    no-relevant-context guardrail.
    """
    lang = lang or config.DATASET_LANG
    split = split or config.DATASET_SPLIT
    path = _resolve_parquet_path(lang, split)

    con = duckdb.connect()
    rows = con.execute(
        f"SELECT query FROM read_parquet('{path}') WHERE query IS NOT NULL LIMIT {n * 2}"
    ).fetchall()
    seen = set()
    queries = []
    for (q,) in rows:
        q = (q or "").strip()
        if q and q not in seen:
            seen.add(q)
            queries.append(q)
        if len(queries) >= n:
            break
    return queries
