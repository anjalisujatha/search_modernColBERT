def search(query: str, top_k: int, model, retriever, conn) -> list[dict]:
    query_embeddings = model.encode(
        [query],
        batch_size=1,
        is_query=True,
        show_progress_bar=False,
    )
    results = retriever.retrieve(queries_embeddings=query_embeddings, k=top_k)
    cursor = conn.cursor()
    hits = []
    for hit in results[0]:
        cursor.execute("SELECT product_title FROM products WHERE pid = ?", (hit["id"],))
        row = cursor.fetchone()
        if row is None:
            continue
        hits.append({"score": round(hit["score"], 4), "title": row[0]})
    return hits
