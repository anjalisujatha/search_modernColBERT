# Using ModernColBERT for implementing semantic search on Amazon ESCI dataset

Semantic search pipeline built on [PyLate](https://github.com/lightonai/pylate) and the `lightonai/colbertv2.0` ColBERT model, indexed over the [Amazon ESCI](https://github.com/amazon-science/esci-data) product dataset.

## Setup

```bash
make setup
```

This creates a `.venv` virtual environment and installs all dependencies (core, notebook, and dev) from `pyproject.toml`.

## API

Start the server:

```bash
.venv/bin/uvicorn main:app --reload
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

## Pipeline

The database and index are already included — just start the API server and query it.

---

### Re-indexing (only needed when indexing more than 20k documents)

The default index covers 20k documents. To index a larger slice or the full dataset, run the full pipeline:

**1. Prepare data**


Run:

```bash
.venv/bin/python scripts/prepare_data.py
```

**2. Build the index**

Adjust `LIMIT` in `index_data.py` (set to `None` for the full dataset), then run:

```bash
.venv/bin/python scripts/index_data.py
```

Encodes all products using the ColBERT model and adds their embeddings to a Voyager index. The finished index is compressed to `data/processed/pylate_index.tar.gz`.

## Tests

```bash
make test
```

## Project structure

```
scripts/
  prepare_data.py   # load, clean, and store products in SQLite
  index_data.py     # encode products and build Voyager index
  retrieve_data.py  # encode query and retrieve top-K results
  utils.py          # shared paths, device selection, index helpers
data/
  raw/              # source parquet (not committed)
  processed/        # SQLite db and compressed index (tracked via Git LFS)
tests/
  test_index_data.py
  test_retrieve_data.py
```