import shutil
import sqlite3
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import ollama
import pandas as pd
import torch
from pylate import indexes, models, retrieve
from sqlitedict import SqliteDict
from tqdm.auto import tqdm

from src.search import EXPAND_N, ORIGINAL_WEIGHT, retrieve_top_pids

K = 5
N_SAMPLE = 100

DATA_DIR = Path(__file__).parent.parent.parent / "data"
INDEX_FOLDER = str(DATA_DIR / "processed" / "pylate_index")
INDEX_NAME = "esci_data_index"
DB_PATH = str(DATA_DIR / "processed" / "products_dataset.db")
PARQUET_PATH = str(DATA_DIR / "raw" / "shopping_queries_dataset_examples.parquet")

RELEVANCE_MAP = {
    "exact": 1.0,
    "substitute": 0.1,
    "complement": 0.01,
    "irrelevant": 0.0,
}
ESCI_CODE_MAP = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
CODE_TO_NAME = {"E": "exact", "S": "substitute", "C": "complement", "I": "irrelevant"}

def setup_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return models.ColBERT(model_name_or_path="lightonai/colbertv2.0", device=device)

def setup_index():
    try:
        index = indexes.Voyager(
            index_folder=INDEX_FOLDER,
            index_name=INDEX_NAME,
            override=False,
        )
    except RuntimeError as e:
        print(f"Warning: Could not load existing index ({e}).\n"
              "The index file is corrupted or missing. Creating a new empty index.\n"
              "Run the indexing notebook (indexing_data.ipynb) on Colab to rebuild it.")
        index = indexes.Voyager(
            index_folder=INDEX_FOLDER,
            index_name=INDEX_NAME,
            override=True,
        )
    index.ef_search = 128
    return index

def setup_ollama():
    try:
        ollama.list()
        print("Ollama already running")
    except ConnectionError:
        ollama_bin = shutil.which("ollama")
        if ollama_bin is None:
            raise RuntimeError(
                "Ollama server is not running and the ollama CLI was not found on PATH.\n"
                "Install it from https://ollama.com, then re-run."
            )
        subprocess.Popen([ollama_bin, "serve"])
        time.sleep(5)
        print("Ollama started")
    ollama.pull("llama3.2")
    print("llama3.2 ready")

def load_eval_queries(index):
    test_df = pd.read_parquet(
        PARQUET_PATH, columns=["query_id", "query", "product_id", "esci_label"]
    )

    doc_map = SqliteDict(
        f"{INDEX_FOLDER}/{INDEX_NAME}/document_ids_to_embeddings.sqlite",
        outer_stack=False,
    )
    indexed_pids = set(int(k) for k in doc_map.keys())
    doc_map.close()

    conn = sqlite3.connect(DB_PATH)
    all_products = pd.read_sql("SELECT pid, product_id FROM products", conn)
    conn.close()

    indexed_asins = set(
        all_products[all_products["pid"].isin(indexed_pids)]["product_id"]
    )

    print(f"Voyager token vectors:   {index.index.num_elements}")
    print(f"Actual indexed docs:     {len(indexed_pids)}")
    print(f"Matching ASINs:          {len(indexed_asins)}")
    print(f"Total products in DB:    {len(all_products)}")
    print(f"Total test rows:         {len(test_df)}")

    test_df_filtered = test_df[test_df["product_id"].isin(indexed_asins)]
    valid_query_ids = test_df_filtered["query_id"].unique()
    print(f"Queries with indexed ground truth: {len(valid_query_ids)}")

    if len(valid_query_ids) == 0:
        raise RuntimeError(
            "No matching queries found — the index is empty or was built from different products.\n"
            "Rebuild the index by running indexing_data.ipynb on Google Colab, "
            "then download pylate_index.tar.gz to data/processed/ and extract it."
        )

    n_sample = min(N_SAMPLE, len(valid_query_ids))
    eval_queries = (
        test_df_filtered.drop_duplicates("query_id").sample(n_sample, random_state=42)
    )
    print(f"Sampled eval queries:    {len(eval_queries)}")
    return eval_queries, test_df_filtered

def retrieve_results(eval_queries, model, retriever):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    evaluation_results = []

    for _, row in tqdm(eval_queries.iterrows(), total=len(eval_queries)):
        q_id = row["query_id"]
        q_text = row["query"]

        retrieved_pids = [pid for pid, _ in retrieve_top_pids(q_text, model, retriever, K)]

        placeholders = ",".join("?" * len(retrieved_pids))
        cursor.execute(
            f"SELECT pid, product_id, product_title FROM products WHERE pid IN ({placeholders})",
            retrieved_pids,
        )
        pid_rows = {r[0]: (r[1], r[2]) for r in cursor.fetchall()}

        evaluation_results.append({
            "query": q_text,
            "query_id": q_id,
            "hits": pid_rows,
            "retrieved_pids": retrieved_pids,
        })

    conn.close()
    return evaluation_results


def attach_labels(evaluation_results, eval_queries, test_df_filtered):
    labeled = []
    for result, (_, eval_row) in zip(evaluation_results, eval_queries.iterrows()):
        q_id = eval_row["query_id"]
        query_ground_truth = test_df_filtered[test_df_filtered["query_id"] == q_id]
        pid_rows = result["hits"]

        hits_with_labels = []
        for rank, pid in enumerate(result["retrieved_pids"]):
            asin, title = pid_rows.get(pid, (None, "Title Not Found"))
            if asin is not None:
                match = query_ground_truth[query_ground_truth["product_id"] == asin]
                if not match.empty:
                    raw_code = match.iloc[0]["esci_label"]
                    label = CODE_TO_NAME.get(raw_code, "irrelevant")
                else:
                    label = "irrelevant"
            else:
                label = "irrelevant"
            hits_with_labels.append({
                "rank": rank + 1,
                "product_id": asin,
                "title": title,
                "label": label,
            })

        labeled.append({
            "query": result["query"],
            "query_id": q_id,
            "hits": hits_with_labels,
        })
    return labeled


# ---------------------------------------------------------------------------
# nDCG scoring
# ---------------------------------------------------------------------------

def get_ndcg(hits, query_id, test_df_filtered, k=5):
    actual_relevance = [RELEVANCE_MAP.get(h["label"].lower(), 0.0) for h in hits[:k]]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(actual_relevance))

    query_gt = test_df_filtered[test_df_filtered["query_id"] == query_id]
    ideal_relevance = sorted(
        [ESCI_CODE_MAP.get(l, 0.0) for l in query_gt["esci_label"]], reverse=True
    )
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance[:k]))

    return dcg / idcg if idcg > 0 else 0.0


def compute_ndcg(evaluation_results, test_df_filtered, k=5):
    scores = [
        get_ndcg(r["hits"], r["query_id"], test_df_filtered, k)
        for r in evaluation_results
    ]
    mean = float(np.mean(scores))
    print(f"\n--- nDCG Evaluation Results ---")
    print(f"Mean nDCG@{k}: {mean:.4f}")
    return mean


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

def judge_relevance(query, product_title):
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a product search relevance judge.\n"
                    "Rate the relevance of a product to a search query using exactly one label:\n"
                    "E - Exact: product directly satisfies the query\n"
                    "S - Substitute: product could substitute for what was searched\n"
                    "C - Complement: product complements but does not match the query\n"
                    "I - Irrelevant: product is not relevant to the query\n"
                    "Output ONLY a single letter: E, S, C, or I"
                ),
            },
            {"role": "user", "content": f"Query: {query}\nProduct: {product_title}"},
        ],
    )
    text = response.message.content.strip().upper()
    for char in text:
        if char in "ESCI":
            return char
    return "I"


def run_llm_evaluation(evaluation_results):
    llm_judged_results = []
    for result in tqdm(evaluation_results, desc="LLM judging"):
        llm_hits = []
        for hit in result["hits"]:
            title = hit["title"]
            llm_code = judge_relevance(result["query"], title) if title and title != "Title Not Found" else "I"
            llm_hits.append({
                **hit,
                "llm_label": llm_code,
                "llm_score": ESCI_CODE_MAP[llm_code],
            })
        llm_judged_results.append({
            "query": result["query"],
            "query_id": result["query_id"],
            "hits": llm_hits,
        })
    return llm_judged_results


def get_llm_ndcg(llm_hits, query_id, test_df_filtered, k=5):
    actual = [h["llm_score"] for h in llm_hits[:k]]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(actual))
    query_gt = test_df_filtered[test_df_filtered["query_id"] == query_id]
    ideal = sorted(
        [ESCI_CODE_MAP.get(l, 0.0) for l in query_gt["esci_label"]], reverse=True
    )
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def compute_llm_evaluation(llm_judged_results, evaluation_results, test_df_filtered, k=5):
    llm_scores = [
        get_llm_ndcg(r["hits"], r["query_id"], test_df_filtered, k)
        for r in llm_judged_results
    ]
    esci_scores = [
        get_ndcg(r["hits"], r["query_id"], test_df_filtered, k)
        for r in evaluation_results
    ]

    print("\n--- LLM-as-Judge vs ESCI Ground Truth ---")
    print(f"nDCG@{k} (ESCI ground truth): {np.mean(esci_scores):.4f}")
    print(f"nDCG@{k} (LLM judge):         {np.mean(llm_scores):.4f}")

    agreed, total = 0, 0
    esci_counts, llm_counts = Counter(), Counter()
    for result in llm_judged_results:
        for hit in result["hits"]:
            esci_label = hit["label"]
            llm_label = CODE_TO_NAME.get(hit["llm_label"], "irrelevant")
            if esci_label == llm_label:
                agreed += 1
            total += 1
            esci_counts[esci_label] += 1
            llm_counts[llm_label] += 1

    print(f"\nLabel agreement (LLM vs ESCI): {agreed}/{total} = {agreed/total:.1%}")
    print(f"\n{'Label':<12} {'ESCI':>6} {'LLM':>6}")
    for label in ["exact", "substitute", "complement", "irrelevant"]:
        print(f"{label:<12} {esci_counts[label]:>6} {llm_counts[label]:>6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Setting up Ollama...")
    setup_ollama()

    print("\nLoading model...")
    model = setup_model()

    print("\nLoading index...")
    index = setup_index()
    retriever = retrieve.ColBERT(index=index)

    print("\nLoading eval queries...")
    eval_queries, test_df_filtered = load_eval_queries(index)

    print(f"\nRunning retrieval (expand_n={EXPAND_N}, original_weight={ORIGINAL_WEIGHT}, K={K})...")
    raw_results = retrieve_results(eval_queries, model, retriever)
    evaluation_results = attach_labels(raw_results, eval_queries, test_df_filtered)

    print("\nExample output (first query):")
    first = evaluation_results[0]
    print(f"Query: {first['query']}")
    for hit in first["hits"]:
        print(f"  Rank {hit['rank']}: {hit['product_id']} | {hit['title']} -> {hit['label']}")

    compute_ndcg(evaluation_results, test_df_filtered, k=K)

    print("\nRunning LLM evaluation...")
    llm_judged_results = run_llm_evaluation(evaluation_results)
    compute_llm_evaluation(llm_judged_results, evaluation_results, test_df_filtered, k=K)


if __name__ == "__main__":
    main()