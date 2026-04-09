# Using ModernColBERT for implementing semantic search on Amazon ESCI dataset

This project explores using [ColBERT](https://arxiv.org/abs/2004.12832) — a late-interaction neural retrieval model — for product search over the [Amazon ESCI dataset](https://github.com/amazon-science/esci-data), a large-scale benchmark of real shopping queries with human-annotated relevance labels.

Unlike traditional keyword search (BM25) or single-vector dense retrieval (e.g. bi-encoders), ColBERT computes token-level interactions between query and document at retrieval time. This gives it finer-grained matching while still being scalable via approximate nearest-neighbor indexing.

The pipeline is built on [PyLate](https://github.com/lightonai/pylate) using the `lightonai/colbertv2.0` model and a Voyager approximate nearest-neighbor index over 50k products. Before retrieval, queries are expanded using `llama3.2` via [Ollama](https://ollama.com) to improve recall on underspecified or ambiguous queries.

## Evaluation Results

We evaluated retrieval quality on 100 queries sampled from the ESCI dataset using two complementary methods:

1. **ESCI ground truth (nDCG@5)** — each retrieved product is scored using the ESCI relevance labels (`exact`, `substitute`, `complement`, `irrelevant`) mapped to a 4-point graded scale, and nDCG@5 is computed against those labels.
2. **LLM-as-judge (nDCG@5)** — `llama3.2` independently assigns a relevance label to each (query, product) pair, and nDCG@5 is computed from those LLM-assigned labels. This tests whether the model's judgement aligns with the human-curated ESCI annotations.

| Metric | Score |
|--------|-------|
| nDCG@5 (ESCI ground truth) | 0.6148 |
| nDCG@5 (LLM judge) | 0.5538 |
| Label agreement (LLM vs ESCI) | 42.2% (211/500) |

**Label distribution across 500 (query, product) pairs:**

| Label | ESCI | LLM |
|-------|------|-----|
| exact | 233 | 172 |
| substitute | 34 | 209 |
| complement | 3 | 8 |
| irrelevant | 230 | 111 |

The LLM tends to over-predict `substitute` and under-predict `irrelevant` compared to ESCI, which explains the lower nDCG score — it is more lenient about what counts as a near-match.

---

## Setup

```bash
make setup
```

This creates a `.venv` virtual environment and installs all dependencies (core, notebook, and dev) from `pyproject.toml`.

## Data

The `data/` folder is not committed to the repository. Download it from Google Drive and add it to the root of this project:

```bash
make download-data
```

## API

Requires [Ollama](https://ollama.com) to be installed and running with the `llama3.2` model pulled:

```bash
ollama pull llama3.2
```

Start the server:

```bash
.venv/bin/uvicorn src.api:app --reload
```

### Endpoints

#### `GET /search/{query}`

Returns the top matching products for a search query.

| Parameter | Type   | Required | Default | Description                  |
|-----------|--------|----------|---------|------------------------------|
| `query`   | string | Yes      | —       | Search query (path parameter)|
| `count`   | int    | No       | 5       | Number of results to return  |

**Example requests:**

```
GET /search/energy efficient fan
GET /search/energy efficient fan?count=10
```

**Example response:**

```json
[
  {"score": 0.8412, "title": "Energy Star Certified Ceiling Fan"},
  {"score": 0.8201, "title": "Quiet Energy Saving Tower Fan"},
  {"score": 0.7985, "title": "Solar Powered Outdoor Fan"}
]
```

---

## Evaluation

Runs nDCG@5 scoring against ESCI ground-truth labels and an LLM-as-judge evaluation using `llama3.2`. Requires the raw parquet file at `data/raw/shopping_queries_dataset_examples.parquet` (included in the Google Drive download).

```bash
.venv/bin/python -m src.scripts.evaluation
```

---

## Re-indexing (only needed when indexing more than 50k documents)

The default index covers 50k documents. To index a larger slice or the full dataset, run the full pipeline:

**1. Prepare data**

```bash
.venv/bin/python src/scripts/prepare_data.py
```

**2. Build the index**

Adjust `LIMIT` in `index_data.py` (set to `None` for the full dataset), then run:

```bash
.venv/bin/python src/scripts/index_data.py
```

Encodes all products using the ColBERT model and builds a Voyager index at `data/processed/pylate_index`.

## Tests

```bash
make test
```

## Project structure

```
src/
  api.py            # FastAPI app
  search.py         # query expansion and retrieval
  scripts/
    prepare_data.py # load, clean, and store products in SQLite
    index_data.py   # encode products and build Voyager index
    evaluation.py   # nDCG and LLM-as-judge evaluation
    utils.py        # shared paths, device selection, index helpers
data/
  raw/              # source parquet
  processed/        # SQLite db and Voyager index
tests/
  test_index_data.py
  test_retrieve_data.py
```