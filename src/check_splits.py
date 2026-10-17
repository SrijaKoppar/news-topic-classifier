"""Phase 3: build and verify leakage-safe splits.

Usage:
    python -m src.check_splits
        python -m src.check_splits --seed 7
    python -m src.check_splits --seed 7
"""
import argparse
import sys

import pandas as pd

from src.config import DATA_PATH, RANDOM_SEED, SPLIT_COLUMNS, TARGET_COL, VALIDATION_FRACTION
from src.data import assert_no_leakage, load_raw, make_split


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--val-fraction", type=float, default=VALIDATION_FRACTION)
    args = ap.parse_args()

    df = load_raw(args.data)
    print(f"seed={args.seed}  val_fraction={args.val_fraction}\n")

    for col in SPLIT_COLUMNS:
        sd = make_split(df, col, seed=args.seed, val_fraction=args.val_fraction)
        assert_no_leakage(df, sd)

        train_texts = set(sd.train["text"].str.lower())
        dup_val = int(sd.val["text"].str.lower().isin(train_texts).sum())
        dup_test = int(sd.test["text"].str.lower().isin(train_texts).sum())
        empty = {n: int((f["text"] == "").sum())
                 for n, f in (("train", sd.train), ("val", sd.val), ("test", sd.test))}

        print(f"=== {col} ===")
        print(f"rows: train={len(sd.train)}  val={len(sd.val)}  test={len(sd.test)}")
        print("leakage checks: PASSED (train/val/test disjoint; "
              "train+val == rows marked train; test == rows marked test)")
        print(f"val rows whose text also appears in train : {dup_val}")
        print(f"test rows whose text also appears in train: {dup_test}")
        print(f"rows with empty combined text: {empty}")
        share = pd.DataFrame({
            "train_%": 100 * sd.train[TARGET_COL].value_counts(normalize=True),
            "val_%": 100 * sd.val[TARGET_COL].value_counts(normalize=True),
        }).round(2)
        print(share.to_string(), "\n")


if __name__ == "__main__":
    main()