import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Mock heavy dependencies before importing the module under test
for _mod in ("torch", "torch.backends", "torch.backends.mps", "pylate", "pylate.indexes", "pylate.models", "pylate.retrieve"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


def _make_db(tmp_path, rows):
    db = tmp_path / "products.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (pid TEXT, product_title TEXT)")
    conn.executemany("INSERT INTO products VALUES (?,?)", rows)
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def mock_model():
    m = MagicMock()
    m.encode.return_value = [np.zeros(128)]
    return m


@pytest.fixture
def mock_retriever():
    return MagicMock()


@pytest.fixture(autouse=True)
def patch_index_lifecycle():
    with (
        patch("retrieve_data.extract_index"),
        patch("retrieve_data.cleanup_index"),
    ):
        yield


def test_query_is_encoded_with_is_query_true(tmp_path, mock_model, mock_retriever):
    """model.encode is called with is_query=True so the query uses the query tower."""
    from retrieve_data import main

    db = _make_db(tmp_path, [("p1", "Fan")])
    mock_retriever.retrieve.return_value = [[{"id": "p1", "score": 0.9}]]

    with (
        patch("retrieve_data.DB_PATH", db),
        patch("retrieve_data.QUERY", "energy efficient fan"),
        patch("retrieve_data.TOP_K", 1),
        patch("retrieve_data.models.ColBERT", return_value=mock_model),
        patch("retrieve_data.indexes.Voyager", return_value=MagicMock()),
        patch("retrieve_data.retrieve.ColBERT", return_value=mock_retriever),
    ):
        main()

    mock_model.encode.assert_called_once()
    assert mock_model.encode.call_args.kwargs["is_query"] is True


def test_retrieve_maps_hits_to_correct_db_titles(tmp_path, mock_model, mock_retriever, capsys):
    """Retriever hits are looked up in SQLite and the matching product titles are printed."""
    from retrieve_data import main

    db = _make_db(tmp_path, [("p1", "Energy Efficient Fan"), ("p2", "Desk Lamp"), ("p3", "USB Hub")])
    mock_retriever.retrieve.return_value = [[{"id": "p1", "score": 0.95}, {"id": "p3", "score": 0.80}]]

    with (
        patch("retrieve_data.DB_PATH", db),
        patch("retrieve_data.QUERY", "fan"),
        patch("retrieve_data.TOP_K", 2),
        patch("retrieve_data.models.ColBERT", return_value=mock_model),
        patch("retrieve_data.indexes.Voyager", return_value=MagicMock()),
        patch("retrieve_data.retrieve.ColBERT", return_value=mock_retriever),
    ):
        main()

    out = capsys.readouterr().out
    assert "Energy Efficient Fan" in out
    assert "USB Hub" in out
    assert "Desk Lamp" not in out  # p2 was not in the retriever results