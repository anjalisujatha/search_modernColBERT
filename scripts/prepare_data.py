import sqlite3

import pandas as pd
from sqlalchemy import create_engine, types

from utils import DB_PATH, ROOT

# --- Config ---
RAW_PARQUET = ROOT / "data/raw/shopping_queries_dataset_products.parquet"
TSV_PATH = ROOT / "data/processed/collection.tsv"
LOCALE = "us"
FRACTION = 3  # use 1/FRACTION of the US dataset (set to 1 for the full dataset)


def load_and_filter(parquet_path: str, locale: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    return df[df["product_locale"] == locale].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Fill nulls
    for col in ["product_bullet_point", "product_brand", "product_color", "product_description"]:
        df[col] = df[col].fillna("")

    # Normalise whitespace
    for col in ["product_title", "product_brand", "product_color", "product_bullet_point"]:
        df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()

    return df


def build_product_text(df: pd.DataFrame) -> pd.DataFrame:
    df["product_text"] = (
        "Title: " + df["product_title"]
        + " | Brand: " + df["product_brand"]
        + " | Color: " + df["product_color"]
        + " | Specs: " + df["product_bullet_point"]
    )
    df["product_text"] = df["product_text"].str.replace(r"[\t\n\r]+", " ", regex=True)
    return df


def save_to_sqlite(df: pd.DataFrame, db_path: str) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    df_save = df.reset_index(drop=True)
    df_save.index.name = "pid"
    df_save.to_sql(
        name="products",
        con=engine,
        if_exists="replace",
        index=True,
        chunksize=10_000,
        dtype={"pid": types.Integer()},
    )
    print(f"Saved {len(df_save)} rows to {db_path}")


def export_tsv(db_path, tsv_path) -> None:
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query("SELECT pid, product_text FROM products", conn)
    conn.close()
    df.to_csv(tsv_path, sep="\t", header=False, index=False)
    print(f"Exported {len(df)} rows to {tsv_path}")


def main():
    print("Loading parquet...")
    df = load_and_filter(RAW_PARQUET, LOCALE)
    print(f"  {len(df)} {LOCALE!r} products loaded")

    df = clean(df)
    df = build_product_text(df)
    df = df.drop(columns=["product_locale"])

    if FRACTION > 1:
        half = len(df) // FRACTION
        df = df.iloc[:half]
        print(f"  Using {len(df)} rows (1/{FRACTION} of dataset)")

    save_to_sqlite(df, DB_PATH)
    export_tsv(DB_PATH, TSV_PATH)


if __name__ == "__main__":
    main()