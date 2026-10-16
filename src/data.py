"""Data loading, validation and text-building utilities."""
"""Data loading, validation, text-building and split utilities."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH, LABELS, RANDOM_SEED, REQUIRED_COLUMNS,
    SPLIT_COLUMNS, TARGET_COL, VALIDATION_FRACTION,
)


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Read the CSV as strings; only truly empty cells become missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put news.csv in the data/ folder."
        )
    df = pd.read_csv(
        path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8"
    )
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"news.csv is missing required columns: {missing}")


def clean_text(series: pd.Series) -> pd.Series:
    """Missing -> '', trim whitespace. Blank and missing become identical."""
    return series.fillna("").astype(str).str.strip()


def combine_text(headline, short_description) -> str:
    """The single model input. Used in training AND by the API.

    Safe with None/NaN. Joins non-empty parts with one space and
    collapses repeated whitespace.
    """
    parts = []
    for part in (headline, short_description):
        if isinstance(part, str) and part.strip():
            parts.append(part.strip())
    return " ".join(" ".join(parts).split())


def unknown_labels(df: pd.DataFrame) -> pd.Series:
    """Counts of category values that are not one of the 9 allowed labels."""
    return df.loc[~df[TARGET_COL].isin(LABELS), TARGET_COL].value_counts(dropna=False)

# --- Leakage-safe train / validation / test construction --------------------

@dataclass(frozen=True)
class SplitData:
    """The three row groups for ONE split column.

    train : used to fit models and tune settings
    val   : used to choose settings and the abstention threshold
    test  : untouched until the single final evaluation (Phase 6)
    """
    split_col: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def make_split(df, split_col, seed=RANDOM_SEED, val_fraction=VALIDATION_FRACTION) -> SplitData:
    """Carve a stratified validation set out of split_col's TRAIN rows only."""
    if split_col not in SPLIT_COLUMNS:
        raise ValueError(f"split_col must be one of {SPLIT_COLUMNS}, got {split_col!r}")
    bad = set(df[split_col].dropna().unique()) - {"train", "test"}
    if bad or df[split_col].isna().any():
        raise ValueError(f"{split_col} must contain only 'train'/'test'; found {bad or 'missing'}")
    if df["id"].duplicated().any():
        raise ValueError("Duplicate ids: cannot verify leakage safely by id.")

    df = df.copy()
    df["text"] = [
        combine_text(h, d) for h, d in zip(df["headline"], df["short_description"])
    ]
    train_full = df[df[split_col] == "train"]
    test = df[df[split_col] == "test"]

    train, val = train_test_split(
        train_full,
        test_size=val_fraction,
        stratify=train_full[TARGET_COL],
        random_state=seed,
    )
    return SplitData(
        split_col=split_col,
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_no_leakage(df, sd: SplitData) -> None:
    """Prove train/val/test are disjoint and match the original split labels."""
    col = sd.split_col
    train_ids, val_ids, test_ids = set(sd.train["id"]), set(sd.val["id"]), set(sd.test["id"])
    marked_train = set(df.loc[df[col] == "train", "id"])
    marked_test = set(df.loc[df[col] == "test", "id"])

    _check(not (train_ids & val_ids), "train and val share rows")
    _check(not (train_ids & test_ids), "train and test share rows")
    _check(not (val_ids & test_ids), "val and test share rows")
    _check(train_ids | val_ids == marked_train, f"train+val != rows marked train in {col}")
    _check(test_ids == marked_test, f"test != rows marked test in {col}")