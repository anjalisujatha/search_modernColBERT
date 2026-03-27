import shutil
import tarfile
from pathlib import Path

import torch

# --- Shared paths ---
ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data/processed/products_dataset.db"
INDEX_DIR = ROOT / "data/processed/pylate_index"
ARCHIVE = ROOT / "data/processed/pylate_index.tar.gz"
INDEX_NAME = "esci_data_index"
MODEL_NAME = "lightonai/colbertv2.0"


def get_device() -> str:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def remove_stale_archive() -> None:
    if ARCHIVE.exists():
        ARCHIVE.unlink()
        print(f"Deleted existing archive: {ARCHIVE}")


def compress_index() -> None:
    print(f"Compressing {INDEX_DIR} -> {ARCHIVE} ...")
    try:
        with tarfile.open(ARCHIVE, "w:gz") as tar:
            tar.add(INDEX_DIR, arcname=INDEX_DIR.name)
    except tarfile.TarError as e:
        raise RuntimeError(f"Failed to compress index to {ARCHIVE}: {e}") from e
    shutil.rmtree(INDEX_DIR)
    print(f"Compressed index saved to {ARCHIVE} ({ARCHIVE.stat().st_size / 1e6:.1f} MB). Uncompressed folder deleted.")


def extract_index() -> None:
    if not INDEX_DIR.exists():
        print(f"Extracting {ARCHIVE} ...")
        try:
            with tarfile.open(ARCHIVE, "r:gz") as tar:
                tar.extractall(path=INDEX_DIR.parent)
        except tarfile.TarError as e:
            raise RuntimeError(f"Failed to extract index from {ARCHIVE}: {e}") from e
        print("Extraction complete.")


def cleanup_index() -> None:
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
        print(f"Deleted extracted index: {INDEX_DIR}")