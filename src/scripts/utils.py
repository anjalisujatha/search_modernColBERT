from pathlib import Path

import torch

# --- Shared paths ---
ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data/processed/products_dataset.db"
INDEX_DIR = ROOT / "data/processed/pylate_index"
INDEX_NAME = "esci_data_index"
MODEL_NAME = "lightonai/colbertv2.0"


def get_device() -> str:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def check_index() -> None:
    if not INDEX_DIR.exists():
        raise RuntimeError(
            f"Index not found at {INDEX_DIR}.\n"
            "Download the data folder from Google Drive and place it at the project root."
            "Refer README.md for more details."
        )