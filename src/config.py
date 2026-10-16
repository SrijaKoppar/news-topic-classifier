"""Central configuration: paths, random seed, split names, labels.

Every other module imports from here so these values are defined once.
"""
import random
from pathlib import Path

import numpy as np

# --- Paths -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "news.csv"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# --- Reproducibility -------------------------------------------------------
RANDOM_SEED = 42

# --- Data schema -----------------------------------------------------------
SPLIT_COLUMNS = ("split_a", "split_b")  # two independent experiments
TARGET_COL = "category"
TEXT_COLS = ("headline", "short_description")  # the ONLY model inputs

REQUIRED_COLUMNS = (
    "id", "headline", "short_description", "authors",
    "link", "date", "category", "split_a", "split_b",
)

LABELS = (
    "POLITICS",
    "WORLD",
    "BUSINESS & MONEY",
    "TECH & SCIENCE",
    "ENTERTAINMENT",
    "SPORTS",
    "HEALTH & WELLNESS",
    "LIFESTYLE",
    "EDUCATION",
)

# --- Experiment design (used from Phase 3) ---------------------------------
# Fraction of each split's TRAIN rows held out as validation.
VALIDATION_FRACTION = 0.15


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python and NumPy. (Torch is seeded in Phase 5 if needed.)"""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    """Create output folders if they don't exist."""
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("ROOT_DIR   :", ROOT_DIR)
    print("DATA_PATH  :", DATA_PATH, "| exists:", DATA_PATH.exists())
    print("SEED       :", RANDOM_SEED)
    print("SPLITS     :", SPLIT_COLUMNS)
    print("N LABELS   :", len(LABELS))