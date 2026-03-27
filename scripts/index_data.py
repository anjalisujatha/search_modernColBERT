import gc
import sqlite3
import time

import torch
from pylate import indexes, models
from tqdm.auto import tqdm

from utils import (
    DB_PATH,
    INDEX_DIR,
    INDEX_NAME,
    MODEL_NAME,
    compress_index,
    get_device,
    remove_stale_archive,
)

# --- Config ---
OVERRIDE_INDEX = True      # set to False to load an existing index
LIMIT = 20000              # set to None to index all documents
ENCODE_BATCH_SIZE = 128    # docs per model.encode() call (GPU memory bound)
INDEX_BATCH_SIZE = 10_000  # docs accumulated before calling add_documents (RAM bound)


def iter_batches(cursor, batch_size):
    while chunk := cursor.fetchmany(batch_size):
        ids, texts = zip(*chunk)
        yield list(ids), list(texts)


def flush_to_index(index, ids, embeddings):
    index.add_documents(documents_ids=ids, documents_embeddings=embeddings)
    ids.clear()
    embeddings.clear()


def main():
    remove_stale_archive()

    device = get_device()

    model = models.ColBERT(model_name_or_path=MODEL_NAME, device=device)

    index = indexes.Voyager(
        index_folder=str(INDEX_DIR),
        index_name=INDEX_NAME,
        override=OVERRIDE_INDEX,
    )

    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()

        if LIMIT:
            cursor.execute("SELECT pid, product_text FROM products LIMIT ?", (LIMIT,))
        else:
            cursor.execute("SELECT pid, product_text FROM products")

        total_start = time.time()
        encoding_time = 0
        indexing_time = 0
        pending_ids = []
        pending_embeddings = []

        n_batches = (LIMIT // ENCODE_BATCH_SIZE) if LIMIT else None
        for batch_ids, batch_texts in tqdm(
            iter_batches(cursor, ENCODE_BATCH_SIZE),
            total=n_batches,
            unit="batch",
            desc="Encoding",
        ):
            t0 = time.time()
            embeddings = model.encode(
                batch_texts,
                is_query=False,
                batch_size=ENCODE_BATCH_SIZE,
                pool_factor=2,
                show_progress_bar=False,
            )
            if isinstance(embeddings, list):
                embeddings = [e.cpu() if hasattr(e, "cpu") else e for e in embeddings]

            if device == "mps":
                torch.mps.empty_cache()

            encoding_time += time.time() - t0

            pending_ids.extend(batch_ids)
            pending_embeddings.extend(embeddings)

            if len(pending_ids) >= INDEX_BATCH_SIZE:
                print(f"Flushing {len(pending_ids)} documents to index...")
                t0 = time.time()
                flush_to_index(index, pending_ids, pending_embeddings)
                indexing_time += time.time() - t0

        if pending_ids:
            print(f"Flushing final {len(pending_ids)} documents to index...")
            t0 = time.time()
            flush_to_index(index, pending_ids, pending_embeddings)
            indexing_time += time.time() - t0

    if device == "mps":
        torch.mps.empty_cache()
    gc.collect()

    print(f"Encoding time: {encoding_time:.1f}s")
    print(f"Indexing time: {indexing_time:.1f}s")
    print(f"Total time:    {time.time() - total_start:.1f}s")

    compress_index()


if __name__ == "__main__":
    main()