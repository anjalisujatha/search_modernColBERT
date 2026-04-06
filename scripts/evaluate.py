import argparse
import sqlite3

import numpy as np
import pandas as pd
import torch
from pylate import indexes, models, retrieve
from sqlitedict import SqliteDict
from tqdm.auto import tqdm

# ESCI relevance mappings
RELEVANCE_MAP = {
    "exact": 1.0,
    "substitute": 0.1,
    "complement": 0.01,
    "irrelevant": 0.0,
}
ESCI_CODE_MAP = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
CODE_TO_NAME = {"E": "exact", "S": "substitute", "C": "complement", "I": "irrelevant"}


def load_index(index_folder, index_name, ef_search=128):
    index = indexes.Voyager(
        index_folder=index_folder,
        index_name=index_name,
        override=False,
    )
    index.ef_search = ef_search
    retriever = retrieve.ColBERT(index=index)
    return index, retriever


def get_indexed_asins(index, db_path):
    """Return the set of ASINs actually present in the Voyager index."""
    doc_map = SqliteDict(
        f"{index.documents_ids_to_embeddings_path}",
        outer_stack=False,
    )
    indexed_pids = {int(k) for k in doc_map.keys()}
    doc_map.close()

    conn = sqlite3.connect(db_path)
    all_products = pd.read_sql("SELECT pid, product_id FROM products", conn)
    conn.close()

    indexed_asins = set(
        all_products[all_products["pid"].isin(indexed_pids)]["product_id"]
    )
    print(f"Voyager token vectors : {index.index.num_elements}")
    print(f"Actual indexed docs   : {len(indexed_pids)}")
    print(f"Matching ASINs        : {len(indexed_asins)}")
    return indexed_asins


def build_eval_queries(examples_path, indexed_asins, n_queries=100, random_state=42):
    """Filter ESCI examples to indexed products and sample eval queries."""
    test_df = pd.read_parquet(
        examples_path, columns=["query_id", "query", "product_id", "esci_label"]
    )
    test_df_filtered = test_df[test_df["product_id"].isin(indexed_asins)]
    valid_query_ids = test_df_filtered["query_id"].unique()

    print(f"Filtered test pairs   : {len(test_df_filtered)}")
    print(f"Queries with GT       : {len(valid_query_ids)}")

    eval_queries = (
        test_df_filtered.drop_duplicates("query_id").sample(
            n_queries, random_state=random_state
        )
    )
    print(f"Sampled eval queries  : {len(eval_queries)}")
    return test_df_filtered, eval_queries


def run_retrieval(model, retriever, eval_queries, test_df_filtered, db_path, k=5):
    """Retrieve top-K products for each query and attach ESCI labels."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    evaluation_results = []

    for _, row in tqdm(eval_queries.iterrows(), total=len(eval_queries)):
        q_id = row["query_id"]
        q_text = row["query"]

        query_emb = model.encode(
            [q_text], is_query=True, batch_size=1, show_progress_bar=False
        )
        search_hits = retriever.retrieve(query_emb, k=k)[0]
        retrieved_pids = [hit["id"] for hit in search_hits]

        placeholders = ",".join("?" * len(retrieved_pids))
        cursor.execute(
            f"SELECT pid, product_id, product_title FROM products WHERE pid IN ({placeholders})",
            retrieved_pids,
        )
        pid_rows = {r[0]: (r[1], r[2]) for r in cursor.fetchall()}

        query_ground_truth = test_df_filtered[test_df_filtered["query_id"] == q_id]

        hits_with_labels = []
        for rank, pid in enumerate(retrieved_pids):
            asin, title = pid_rows.get(pid, (None, "Title Not Found"))
            if asin is not None:
                match = query_ground_truth[query_ground_truth["product_id"] == asin]
                label = (
                    CODE_TO_NAME.get(match.iloc[0]["esci_label"], "irrelevant")
                    if not match.empty
                    else "irrelevant"
                )
            else:
                label = "irrelevant"

            hits_with_labels.append(
                {"rank": rank + 1, "product_id": asin, "title": title, "label": label}
            )

        evaluation_results.append({"query": q_text, "query_id": q_id, "hits": hits_with_labels})

    conn.close()
    return evaluation_results


def get_ndcg(hits, query_id, test_df_filtered, k=5):
    actual_relevance = [RELEVANCE_MAP.get(h["label"].lower(), 0.0) for h in hits[:k]]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(actual_relevance))

    query_gt = test_df_filtered[test_df_filtered["query_id"] == query_id]
    ideal_relevance = sorted(
        [ESCI_CODE_MAP.get(l, 0.0) for l in query_gt["esci_label"]], reverse=True
    )
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance[:k]))

    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(evaluation_results, test_df_filtered, k=5):
    scores = [
        get_ndcg(r["hits"], r["query_id"], test_df_filtered, k=k)
        for r in evaluation_results
    ]
    mean_ndcg = np.mean(scores)
    print(f"\n--- Evaluation Results ---")
    print(f"Mean nDCG@{k}: {mean_ndcg:.4f}")
    return mean_ndcg


def main():
    parser = argparse.ArgumentParser(description="Evaluate ModernColBERT retrieval")
    parser.add_argument("--index-folder", default="data/processed/pylate_index")
    parser.add_argument("--index-name", default="esci_data_index")
    parser.add_argument("--db-path", default="data/processed/products_dataset.db")
    parser.add_argument(
        "--examples-path",
        default="data/raw/shopping_queries_dataset_examples.parquet",
    )
    parser.add_argument("--model-name", default="lightonai/colbertv2.0")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ef-search", type=int, default=128)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading model...")
    model = models.ColBERT(model_name_or_path=args.model_name, device=device)

    print("\nLoading index...")
    index, retriever = load_index(args.index_folder, args.index_name, args.ef_search)

    print("\nBuilding eval set...")
    indexed_asins = get_indexed_asins(index, args.db_path)
    test_df_filtered, eval_queries = build_eval_queries(
        args.examples_path, indexed_asins, args.n_queries, args.random_state
    )

    print("\nRunning retrieval...")
    evaluation_results = run_retrieval(
        model, retriever, eval_queries, test_df_filtered, args.db_path, k=args.k
    )

    compute_metrics(evaluation_results, test_df_filtered, k=args.k)


if __name__ == "__main__":
    main()