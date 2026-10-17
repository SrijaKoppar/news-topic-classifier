"""Phase 2: explore news.csv and write a human-readable report.

Usage:
    python -m src.explore
        python -m src.explore --out results/my_report.md
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_PATH, LABELS, REQUIRED_COLUMNS, RESULTS_DIR,
    SPLIT_COLUMNS, TARGET_COL, ensure_dirs,
)
from src.data import clean_text, combine_text, load_raw, unknown_labels

PCTS = [0.01, 0.05, 0.5, 0.95, 0.99]


def block(obj, max_colwidth=None) -> str:
    if isinstance(obj, pd.DataFrame):
        text = obj.to_string(max_colwidth=max_colwidth)
    elif hasattr(obj, "to_string"):
        text = obj.to_string()
    else:
        text = str(obj)
    return "```\n" + text + "\n```\n"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns (prefixed '_'): cleaned parts, model text, dedup key."""
    df = df.copy()
    df["_h"] = clean_text(df["headline"])
    df["_d"] = clean_text(df["short_description"])
    df["_text"] = [combine_text(h, d) for h, d in zip(df["_h"], df["_d"])]
    df["_key"] = df["_text"].str.lower()
    return df


def section_overview(df) -> str:
    out = [f"- Rows: {len(df)}", f"- Columns: {list(df.columns[: len(df.columns) - 4])}"]
    extra = [c for c in df.columns if not c.startswith("_") and c not in REQUIRED_COLUMNS]
    out.append(f"- Unexpected extra columns: {extra if extra else 'none'}")
    bad = unknown_labels(df)
    out.append("- Labels outside the allowed 9: " + ("none" if bad.empty else ""))
    text = "\n".join(out) + "\n"
    if not bad.empty:
        text += block(bad)
    absent = set(LABELS) - set(df[TARGET_COL].dropna())
    text += f"- Allowed labels that never appear: {sorted(absent) if absent else 'none'}\n"
    for col in SPLIT_COLUMNS:
        text += f"\nValues in `{col}` (including missing):\n" + block(
            df[col].value_counts(dropna=False)
        )
    return text


def section_missing(df) -> str:
    rows = []
    for col in REQUIRED_COLUMNS:
        n = int((clean_text(df[col]) == "").sum())
        rows.append({"column": col, "missing_or_blank": n,
                     "pct": round(100 * n / len(df), 2)})
    table = pd.DataFrame(rows).set_index("column")
    both = int(((df["_h"] == "") & (df["_d"] == "")).sum())
    return block(table) + f"\nRows with BOTH headline and description empty: {both}\n"


def section_duplicates(df) -> str:
    nonempty = df[df["_key"] != ""]
    content_cols = [c for c in df.columns if not c.startswith("_") and c != "id"]
    out = [
        f"- Extra rows sharing an `id`: {int(df['id'].duplicated().sum())}",
        f"- Extra rows identical in every column except `id`: "
        f"{int(df[content_cols].duplicated().sum())}",
        f"- Extra rows with identical model text (case-insensitive): "
        f"{int(nonempty['_key'].duplicated().sum())}",
    ]
    labels_per_text = nonempty.groupby("_key")[TARGET_COL].nunique()
    out.append(f"- Distinct texts that appear with more than one label: "
               f"{int((labels_per_text > 1).sum())}")
    text = "\n".join(out) + "\n"

    for col in SPLIT_COLUMNS:
        train = nonempty[nonempty[col] == "train"]
        test = nonempty[nonempty[col] == "test"]
        train_labels = train.groupby("_key")[TARGET_COL].agg(set)
        overlap = test[test["_key"].isin(train_labels.index)]
        disagree = sum(
            lab not in train_labels[key]
            for key, lab in zip(overlap["_key"], overlap[TARGET_COL])
        )
        text += (
            f"\n`{col}`: {len(overlap)} of {len(test)} test rows have text that also "
            f"appears in this split's train rows; {disagree} of those carry a label "
            f"never seen for that text in train.\n"
        )
    return text


def section_class_balance(df) -> str:
    counts = df[TARGET_COL].value_counts().reindex(list(LABELS)).fillna(0).astype(int)
    overall = pd.DataFrame({"count": counts, "pct": (100 * counts / counts.sum()).round(2)})
    text = block(overall)
    nz = counts[counts > 0]
    if len(nz) > 1:
        text += f"\nLargest / smallest class ratio: {nz.max() / nz.min():.1f}\n"
    for col in SPLIT_COLUMNS:
        ct = pd.crosstab(df[TARGET_COL], df[col])
        pct = (100 * ct / ct.sum()).round(2).add_suffix("_%")
        text += f"\nClass balance by `{col}`:\n" + block(pd.concat([ct, pct], axis=1))
    return text


def section_splits(df) -> str:
    text = ""
    for col in SPLIT_COLUMNS:
        text += f"`{col}` counts:\n" + block(df[col].value_counts(dropna=False))
    text += "\nCross-tab of split_a (rows) vs split_b (columns):\n"
    text += block(pd.crosstab(df[SPLIT_COLUMNS[0]], df[SPLIT_COLUMNS[1]]))
    return text


def section_lengths(df) -> str:
    stats = {}
    for name, s in {"headline": df["_h"], "short_description": df["_d"],
                    "combined": df["_text"]}.items():
        stats[f"{name} chars"] = s.str.len().describe(percentiles=PCTS)
        stats[f"{name} words"] = s.str.split().str.len().describe(percentiles=PCTS)
    return block(pd.DataFrame(stats).T.round(1))


def section_dates(df) -> str:
    parsed = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    present = clean_text(df["date"]) != ""
    unparseable = int((parsed.isna() & present).sum())
    out = (
        f"- Missing dates: {int((~present).sum())}\n"
        f"- Present but not YYYY-MM-DD: {unparseable}\n"
        f"- Earliest: {parsed.min()}  Latest: {parsed.max()}\n"
    )
    years = parsed.dt.year
    out += "\nRows per year:\n" + block(years.value_counts().sort_index())
    for col in SPLIT_COLUMNS:
        out += f"\nTrain/test date range for `{col}`:\n"
        out += block(parsed.groupby(df[col]).agg(["min", "max"]))
        out += f"\nRows per year by `{col}`:\n" + block(pd.crosstab(years, df[col]))
    share = pd.crosstab(years, df[TARGET_COL], normalize="index").mul(100).round(1)
    out += "\nClass share (%) per year (rows = labels, columns = years):\n"
    out += block(share.T)
    return out


def section_unusual(df) -> str:
    h, d, t = df["_h"], df["_d"], df["_text"]
    words = t.str.split().str.len()
    checks = {
        "combined text under 5 words": int((words < 5).sum()),
        "short_description empty": int((d == "").sum()),
        "headline identical to description": int(((h != "") & (h.str.lower() == d.str.lower())).sum()),
        "text contains a URL": int(t.str.contains(r"https?://|www\.", case=False, regex=True).sum()),
        "text contains HTML tag/entity": int(t.str.contains(
            r"<[a-z/][^>]*>|&[a-z]+;|&#\d+;", case=False, regex=True).sum()),
        "headline ALL CAPS (10+ chars)": int(((h.str.len() >= 10) & (h == h.str.upper())
                                              & h.str.contains(r"[A-Z]", regex=True)).sum()),
    }
    text = block(pd.Series(checks, name="rows"))
    shortest = df.loc[words.nsmallest(10).index, ["_h", "_d", TARGET_COL]]
    shortest.columns = ["headline", "short_description", TARGET_COL]
    text += "\nThe 10 shortest texts (for you to eyeball):\n" + block(shortest, max_colwidth=60)
    return text


SECTIONS = [
    ("1. Overview and schema", section_overview),
    ("2. Missing data", section_missing),
    ("3. Duplicates", section_duplicates),
    ("4. Class balance", section_class_balance),
    ("5. Train/test splits", section_splits),
    ("6. Text lengths", section_lengths),
    ("7. Dates", section_dates),
    ("8. Anything unusual", section_unusual),
]


def build_report(df) -> str:
    parts = ["# Exploration report\n"]
    for title, fn in SECTIONS:
        parts.append(f"## {title}\n")
        parts.append(fn(df))
    return "\n".join(parts)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # headlines may contain non-ASCII
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--out", default=str(RESULTS_DIR / "exploration_report.md"))
    args = ap.parse_args()

    ensure_dirs()
    df = prepare(load_raw(args.data))
    report = build_report(df)
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()