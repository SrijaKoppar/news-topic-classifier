"""Data loading, validation and text-building utilities."""
from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, LABELS, REQUIRED_COLUMNS, TARGET_COL


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