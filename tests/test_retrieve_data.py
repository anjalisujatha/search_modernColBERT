import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock heavy dependencies before importing the module under test
for _mod in ("torch", "torch.backends", "torch.backends.mps", "pylate", "pylate.indexes", "pylate.models", "pylate.retrieve"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from src.search import EXPAND_N  # noqa: E402

EXPANDED_TERMS = [f"expansion_{i}" for i in range(5)]


def _make_db(tmp_path, rows):
    db = tmp_path / "products.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (pid TEXT, product_title TEXT)")
    conn.executemany("INSERT INTO products VALUES (?,?)", rows)
    conn.commit()
    return conn


def _hits_for_queries(*per_query_hits):
    """Build a retriever return value: one hit list per query (original + EXPAND_N expanded)."""
    result = list(per_query_hits)
    result += [[] for _ in range(1 + EXPAND_N - len(per_query_hits))]
    return result


@pytest.fixture
def mock_model():
    m = MagicMock()
    m.encode.return_value = [np.zeros(128)] * (1 + EXPAND_N)
    return m


@pytest.fixture
def mock_retriever():
    return MagicMock()


@pytest.fixture(autouse=True)
def mock_expand(monkeypatch):
    monkeypatch.setattr("src.search.expand_query", lambda q: EXPANDED_TERMS)


def test_query_is_encoded_with_is_query_true(tmp_path, mock_model, mock_retriever):
    """model.encode is called with is_query=True so the query uses the query tower."""
    from src.search import search

    conn = _make_db(tmp_path, [("p1", "Fan")])
    mock_retriever.retrieve.return_value = _hits_for_queries([{"id": "p1", "score": 0.9}])

    search("energy efficient fan", 1, mock_model, mock_retriever, conn)

    mock_model.encode.assert_called_once()
    assert mock_model.encode.call_args.kwargs["is_query"] is True


def test_retrieve_maps_hits_to_correct_db_titles(tmp_path, mock_model, mock_retriever):
    """Retriever hits are looked up in SQLite and the matching product titles are returned."""
    from src.search import search

    conn = _make_db(tmp_path, [("p1", "Energy Efficient Fan"), ("p2", "Desk Lamp"), ("p3", "USB Hub")])
    mock_retriever.retrieve.return_value = _hits_for_queries(
        [{"id": "p1", "score": 0.95}, {"id": "p3", "score": 0.80}]
    )

    results = search("fan", 2, mock_model, mock_retriever, conn)

    titles = [r["title"] for r in results]
    assert "Energy Efficient Fan" in titles
    assert "USB Hub" in titles
    assert "Desk Lamp" not in titles  # p2 was not in the retriever results


def test_missing_pid_is_skipped(tmp_path, mock_model, mock_retriever):
    """Hits whose PID has no matching DB row are skipped rather than crashing."""
    from src.search import search

    conn = _make_db(tmp_path, [("p1", "Fan")])
    mock_retriever.retrieve.return_value = _hits_for_queries(
        [{"id": "p1", "score": 0.9}, {"id": "missing", "score": 0.5}]
    )

    results = search("fan", 2, mock_model, mock_retriever, conn)

    assert len(results) == 1
    assert results[0]["title"] == "Fan"