import sqlite3
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from fastapi import FastAPI
from pylate import indexes, models, retrieve

from utils import DB_PATH, INDEX_DIR, INDEX_NAME, MODEL_NAME, cleanup_index, extract_index, get_device
from retrieve_data import search

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        extract_index()
    except Exception as e:
        raise RuntimeError(f"Failed to extract index from {INDEX_DIR}: {e}") from e

    try:
        device = get_device()
        app.state.model = models.ColBERT(model_name_or_path=MODEL_NAME, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load ColBERT model '{MODEL_NAME}': {e}") from e

    try:
        index = indexes.Voyager(index_folder=str(INDEX_DIR), index_name=INDEX_NAME)
        app.state.retriever = retrieve.ColBERT(index=index)
    except Exception as e:
        raise RuntimeError(f"Failed to load Voyager index '{INDEX_NAME}': {e}") from e

    try:
        app.state.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to database at {DB_PATH}: {e}") from e

    yield
    app.state.conn.close()
    cleanup_index()


app = FastAPI(lifespan=lifespan)


@app.get("/search/{query}")
def get_products(query: str, count: int = 5):
    results = search(query, count, app.state.model, app.state.retriever, app.state.conn)
    return results
