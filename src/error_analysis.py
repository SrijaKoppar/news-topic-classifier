"""Phase 8, step 1: draw a reproducible sample of the locked model's TEST errors.

Reads saved test predictions and joins the text back from news.csv by id.
Does not change any model, threshold or selection.

Usage:
    python -m src.error_analysis sample
    python -m src.error_analysis sample --per-bin 7 --seed 7 --force
"""
import argparse
import json
import sys

import pandas as pd

from src.config import DATA_PATH, RANDOM_SEED, RESULTS_DIR, SPLIT_COLUMNS
from src.data import load_raw

SELECTION_PATH = RESULTS_DIR / "model_selection.json"
OUT_SAMPLE = RESULTS_DIR / "error_analysis_sample.csv"
CONF_BINS = (
    ("high (>=0.80)", 0.80, 1.01),
    ("mid (0.50-0.79)", 0.50, 0.80),
    ("low (<0.50)", 0.00, 0.50),
)


def load_errors(model: str, split_col: str, text: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    d = pd.read_csv(RESULTS_DIR / f"test_{model}_{split_col}_predictions.csv",
                    usecols=["id", "true", "pred", "confidence"], dtype={"id": str})
    d = d.merge(text, on="id", how="left", validate="one_to_one")
    assert d["headline"].notna().all(), "some prediction ids not found in news.csv"
    return d[d["true"] != d["pred"]].copy(), len(d)


def population_summary(errs: pd.DataFrame, n_test: int, split_col: str) -> None:
    high = int((errs["confidence"] >= 0.80).sum())
    print(f"--- {split_col}: {len(errs)} errors out of {n_test} test rows "
          f"({100 * len(errs) / n_test:.1f}%); {high} of them with confidence >= 0.80 "
          f"({100 * high / len(errs):.1f}% of errors)")
    pairs = (errs.groupby(["true", "pred"]).size().sort_values(ascending=False).head(8))
    print("    most common (true -> predicted) confusions:")
    for (t, p), n in pairs.items():
        print(f"      {n:4d}  {t} -> {p}")
    print()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--per-bin", type=int, default=5)
    s.add_argument("--seed", type=int, default=RANDOM_SEED)
    s.add_argument("--force", action="store_true")
    s.add_argument("--data", default=str(DATA_PATH))
    args = ap.parse_args()

    if OUT_SAMPLE.exists() and not args.force:
        raise SystemExit(f"{OUT_SAMPLE.name} already exists (it may hold your annotations). "
                         "Use --force to overwrite.")

    model = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))["common_model"]
    raw = load_raw(args.data)
    text = pd.DataFrame({"id": raw["id"],
                         "headline": raw["headline"].fillna(""),
                         "short_description": raw["short_description"].fillna("")})
    print(f"Model: {model}  (test predictions; sample seed={args.seed})\n")

    parts = []
    for split_col in SPLIT_COLUMNS:
        errs, n_test = load_errors(model, split_col, text)
        population_summary(errs, n_test, split_col)
        for name, lo, hi in CONF_BINS:
            pool = errs[(errs["confidence"] >= lo) & (errs["confidence"] < hi)]
            pick = pool.sample(n=min(args.per_bin, len(pool)), random_state=args.seed)
            pick.insert(0, "conf_bin", name)
            pick.insert(0, "split", split_col)
            parts.append(pick)

    sample = (pd.concat(parts, ignore_index=True)
                .sort_values(["split", "confidence"], ascending=[True, False])
                .reset_index(drop=True))
    sample["confidence"] = sample["confidence"].round(3)
    sample["cause_group"] = ""   # you fill these in
    sample["notes"] = ""
    sample.to_csv(OUT_SAMPLE, index=False)

    for i, r in sample.iterrows():
        print(f"[{i}] {r['split']} | {r['true']} -> {r['pred']} | conf={r['confidence']:.2f} "
              f"| id={r['id']}")
        print(f"    H: {r['headline']}")
        print(f"    D: {(r['short_description'] or '(empty)')[:300]}\n")
    print(f"Saved {len(sample)} examples to {OUT_SAMPLE}")


if __name__ == "__main__":
    main()