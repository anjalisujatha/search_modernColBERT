import ollama

EXPAND_N = 2
ORIGINAL_WEIGHT = 2.0


def expand_query(query: str) -> list[str]:
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search engine optimizer. Provide 5 synonyms or related "
                    "product terms for the user query. Output ONLY the terms separated "
                    "by commas. No explanations."
                ),
            },
            {"role": "user", "content": f"Query: {query}"},
        ],
    )
    return [t.strip() for t in response.message.content.split(",")]


def retrieve_top_pids(query: str, model, retriever, k: int) -> list[tuple[int, float]]:
    """Expand query, encode all variants, retrieve and merge scores.

    Returns a list of (pid, weighted_score) pairs sorted by score descending.
    """
    expanded = expand_query(query)[:EXPAND_N]
    all_queries = [query] + expanded
    query_embs = model.encode(all_queries, is_query=True, show_progress_bar=False)
    raw_hits = retriever.retrieve(query_embs, k=k)

    merged = {}
    for q_idx, query_hits in enumerate(raw_hits):
        weight = ORIGINAL_WEIGHT if q_idx == 0 else 1.0
        for hit in query_hits:
            pid = hit["id"]
            score = hit["score"] * weight
            if pid not in merged or score > merged[pid]:
                merged[pid] = score

    return sorted(merged.items(), key=lambda x: x[1], reverse=True)[:k]


def search(query: str, top_k: int, model, retriever, conn) -> list[dict]:
    top_pids = retrieve_top_pids(query, model, retriever, top_k)
    cursor = conn.cursor()
    hits = []
    for pid, score in top_pids:
        cursor.execute("SELECT product_title FROM products WHERE pid = ?", (pid,))
        row = cursor.fetchone()
        if row is None:
            continue
        hits.append({"score": round(score, 4), "title": row[0]})
    return hits