import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Mock heavy dependencies before importing the module under test.
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.backends", MagicMock())
sys.modules.setdefault("torch.backends.mps", MagicMock())
sys.modules.setdefault("pylate", MagicMock())
sys.modules.setdefault("pylate.indexes", MagicMock())
sys.modules.setdefault("pylate.models", MagicMock())

from index_data import iter_batches, main  # noqa: E402


def _make_db(tmp_path, rows):
    db = tmp_path / "products.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (pid TEXT, product_text TEXT)")
    conn.executemany("INSERT INTO products VALUES (?,?)", rows)
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def mock_model():
    m = MagicMock()
    m.encode.side_effect = lambda texts, **kw: [np.zeros(128)] * len(texts)
    return m


@pytest.fixture
def mock_index():
    return MagicMock()


def test_fetch_data_from_sqlite(tmp_path):
    """iter_batches correctly reads all rows from the products table."""
    rows = [("p1", "blue shirt"), ("p2", "red shoes"), ("p3", "green hat")]
    db = _make_db(tmp_path, rows)

    conn = sqlite3.connect(str(db))
    cursor = conn.cursor()
    cursor.execute("SELECT pid, product_text FROM products")
    batches = list(iter_batches(cursor, batch_size=2))
    conn.close()

    assert batches[0] == (["p1", "p2"], ["blue shirt", "red shoes"])
    assert batches[1] == (["p3"], ["green hat"])


def test_encode_and_create_index(tmp_path, mock_model, mock_index):
    """main encodes each document and adds embeddings to the index."""
    db = _make_db(tmp_path, [("p1", "blue shirt"), ("p2", "red shoes")])
    captured_ids = []
    mock_index.add_documents.side_effect = lambda **kw: captured_ids.extend(kw["documents_ids"])

    with (
        patch("index_data.DB_PATH", db),
        patch("index_data.LIMIT", 2),
        patch("index_data.ENCODE_BATCH_SIZE", 2),
        patch("index_data.INDEX_BATCH_SIZE", 10),
        patch("index_data.models.ColBERT", return_value=mock_model),
        patch("index_data.indexes.Voyager", return_value=mock_index),
        patch("index_data.remove_stale_archive"),
        patch("index_data.compress_index"),
    ):
        main()

    mock_model.encode.assert_called_once()
    mock_index.add_documents.assert_called_once()
    assert captured_ids == ["p1", "p2"]