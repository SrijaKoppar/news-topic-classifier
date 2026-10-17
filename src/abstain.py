"""Phase 7: choose the abstention threshold from VALIDATION predictions only.

Usage:
    python -m src.abstain
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import (
    ABSTAIN_MIN_COVERAGE, ABSTAIN_THRESHOLDS, RESULTS_DIR, SPLIT_COLUMNS,
)

SELECTION_PATH = RESULTS_DIR / "model_selection.json"
OUT_TABLE = RESULTS_DIR / "abstention_threshold_analysis.csv"
OUT_BINS = RESULTS_DIR / "abstention_calibration_bins.csv"
OUT_CHOICE = RESULTS_DIR / "abstention_selection.json"

BIN_EDGES = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
BIN_LABELS = ["<0.30", "0.30-0.39", "0.40-0.49", "0.50-0.59",
              "0.60-0.69", "0.70-0.79", "0.80-0.89", "0.90-1.00"]


def load_val_predictions(model: str, split_col: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{model}_{split_col}_val_predictions.csv"
    return pd.read_csv(path, usecols=["id", "true", "pred", "confidence"])


def analyse_threshold(d: pd.DataFrame, thr: float) -> dict:
    """Answer when confidence >= thr; otherwise abstain."""
    answered = d["confidence"] >= thr
    wrong = d["true"] != d["pred"]
    a = d[answered]
    n, n_ans = len(d), int(answered.sum())
    n_abs = n - n_ans
    n_wrong = int(wrong.sum())
    caught = int((wrong & ~answered).sum())       # errors that abstention avoided
    lost = int((~wrong & ~answered).sum())        # correct answers given up
    labels = sorted(set(a["true"]) | set(a["pred"]))
    return {
        "threshold": thr,
        "coverage": n_ans / n,
        "n_answered": n_ans,
        "n_abstained": n_abs,
        "pct_abstained": 100 * n_abs / n,
        "macro_f1_answered": (f1_score(a["true"], a["pred"], labels=labels,
                                       average="macro", zero_division=0)
                              if n_ans else np.nan),
        "accuracy_answered": float((~wrong[answered]).mean()) if n_ans else np.nan,
        "errors_caught": caught,
        "pct_errors_caught": 100 * caught / n_wrong,
        "correct_lost": lost,
        "pct_correct_lost": 100 * lost / (n - n_wrong),
        "pct_abstained_that_were_wrong": 100 * caught / n_abs if n_abs else np.nan,
    }


def calibration_bins(d: pd.DataFrame, split_col: str) -> pd.DataFrame:
    """Accuracy per confidence bucket: does higher confidence mean more correct?"""
    g = (d.assign(bin=pd.cut(d["confidence"], bins=BIN_EDGES, labels=BIN_LABELS,
                             right=False),
                  correct=d["true"] == d["pred"])
          .groupby("bin", observed=True)
          .agg(n=("correct", "size"), accuracy=("correct", "mean"),
               mean_confidence=("confidence", "mean"))
          .reset_index())
    g["bin"] = g["bin"].astype(str)
    g.insert(0, "split", split_col)
    return g


def choose(table: pd.DataFrame, min_coverage: float):
    """Highest mean macro-F1 among thresholds meeting the coverage floor on BOTH splits."""
    t = table[table["threshold"] > 0]  # exclude the no-abstention reference row
    cov = t.pivot(index="threshold", columns="split", values="coverage")
    f1 = t.pivot(index="threshold", columns="split", values="macro_f1_answered")
    ok = cov.index[cov.min(axis=1) >= min_coverage]
    if len(ok) == 0:
        return None
    return float(f1.loc[ok].mean(axis=1).idxmax())  # ties -> lowest threshold


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    model = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))["common_model"]

    rows, bins = [], []
    for split_col in SPLIT_COLUMNS:
        d = load_val_predictions(model, split_col)
        for thr in (0.0, *ABSTAIN_THRESHOLDS):  # 0.0 = reference: never abstain
            rows.append({"split": split_col, **analyse_threshold(d, thr)})
        bins.append(calibration_bins(d, split_col))
    table, bin_table = pd.DataFrame(rows), pd.concat(bins, ignore_index=True)
    table.to_csv(OUT_TABLE, index=False)
    bin_table.to_csv(OUT_BINS, index=False)

    cols = ["threshold", "coverage", "n_abstained", "pct_abstained",
            "macro_f1_answered", "accuracy_answered",
            "pct_errors_caught", "pct_correct_lost"]
    for split_col in SPLIT_COLUMNS:
        print(f"\n=== {split_col} validation, model={model} (threshold 0.0 = no abstention) ===")
        print(table[table["split"] == split_col][cols].round(3).to_string(index=False))
    print("\n=== Accuracy by confidence bucket (validation) ===")
    print(bin_table.round(3).to_string(index=False))

    chosen = choose(table, ABSTAIN_MIN_COVERAGE)
    if chosen is None:
        raise SystemExit("No threshold keeps coverage >= "
                         f"{ABSTAIN_MIN_COVERAGE:.0%} on both splits. Extend "
                         "ABSTAIN_THRESHOLDS downward or lower ABSTAIN_MIN_COVERAGE.")
    sensitivity = {f"{f:.0%}": choose(table, f) for f in (0.7, 0.8, 0.9)}
    choice = {
        "model": model,
        "threshold": chosen,
        "rule": "highest mean macro-F1 on answered cases, among thresholds with "
                "validation coverage >= the floor on both splits",
        "min_coverage_assumption": ABSTAIN_MIN_COVERAGE,
        "data_used": "validation predictions only (models fitted on train minus validation)",
        "at_threshold": table[table["threshold"] == chosen].round(4).to_dict(orient="records"),
        "threshold_if_coverage_floor_were": sensitivity,
    }
    OUT_CHOICE.write_text(json.dumps(choice, indent=2), encoding="utf-8")
    print(f"\nChosen threshold: {chosen}  (coverage floor {ABSTAIN_MIN_COVERAGE:.0%} on both splits)")
    print(f"Sensitivity to the floor: {sensitivity}")
    print(f"Saved {OUT_TABLE.name}, {OUT_BINS.name}, {OUT_CHOICE.name}")


if __name__ == "__main__":
    main()