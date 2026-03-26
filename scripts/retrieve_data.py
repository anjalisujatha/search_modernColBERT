import sqlite3

from pylate import indexes, models, retrieve

from utils import (
    DB_PATH,
    INDEX_DIR,
    INDEX_NAME,
    MODEL_NAME,
    cleanup_index,
    extract_index,
    get_device,
)

# --- Config ---
QUERY = "energy efficient fan"
TOP_K = 3


def main():
    extract_index()

    index = indexes.Voyager(
        index_folder=str(INDEX_DIR),
        index_name=INDEX_NAME,
    )
    retriever = retrieve.ColBERT(index=index)

    device = get_device()
    model = models.ColBERT(model_name_or_path=MODEL_NAME, device=device)
    query_embeddings = model.encode(
        [QUERY],
        batch_size=1,
        is_query=True,
        show_progress_bar=False,
    )

    results = retriever.retrieve(queries_embeddings=query_embeddings, k=TOP_K)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    print(f"\nTop {TOP_K} results for query: {QUERY!r}")
    for hit in results[0]:
        cursor.execute("SELECT product_title FROM products WHERE pid = ?", (hit["id"],))
        title = cursor.fetchone()[0]
        print(f"  [{hit['score']:.2f}] {title}")

    conn.close()
    cleanup_index()


if __name__ == "__main__":
    main()