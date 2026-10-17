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

    # --- Baseline: TF-IDF + logistic regression (Phase 4) ------------------------
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    lowercase=True,
    strip_accents="unicode",
)
LR_C_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)
LR_CLASS_WEIGHT_GRID = (None, "balanced")
LR_MAX_ITER = 1000

# --- Sentence-embedding model (Phase 5) --------------------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MAX_SEQ_LENGTH = 256   # longer texts are truncated (very few exceed this)
EMBED_BATCH_SIZE = 64
EMBED_C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)

MODEL_NAMES = ("tfidf", "embed")

# --- Abstention (Phase 7) -----------------------------------------------------
ABSTAIN_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
                      0.60, 0.65, 0.70, 0.75, 0.80)
# Assumption: reviewers can handle ~20% of articles, so coverage must stay >= 80%.
ABSTAIN_MIN_COVERAGE = 0.80

# --- Serving (Phase 9) ----------------------------------------------------------
# Assumption: serve the embedding model fitted on ALL split_a train rows.
SERVE_MODEL_DIR = MODELS_DIR / "final" / "embed_split_a"