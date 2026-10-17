"""After-the-fact diagnostic: apply the LOCKED abstention threshold to TEST predictions.

This never changes the threshold. It only reports how the validation-chosen
threshold behaves on the untouched test rows.

Usage:
    python -m src.abstain_diagnostic
"""
import json
import sys

import pandas as pd

from src.abstain import analyse_threshold, calibration_bins
from src.config import RESULTS_DIR, SPLIT_COLUMNS

CHOICE_PATH = RESULTS_DIR / "abstention_selection.json"
OUT_TABLE = RESULTS_DIR / "test_abstention_diagnostic.csv"
OUT_BINS = RESULTS_DIR / "test_abstention_calibration_bins.csv"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    choice = json.loads(CHOICE_PATH.read_text(encoding="utf-8"))
    model, thr = choice["model"], choice["threshold"]
    print(f"Locked from validation: model={model}, threshold={thr}. Not modified here.\n")

    rows, bins = [], []
    for split_col in SPLIT_COLUMNS:
        d = pd.read_csv(RESULTS_DIR / f"test_{model}_{split_col}_predictions.csv",
                        usecols=["id", "true", "pred", "confidence"])
        for t in (0.0, thr):  # 0.0 = no abstention, for comparison
            rows.append({"split": split_col, **analyse_threshold(d, t)})
        bins.append(calibration_bins(d, split_col))

    table, bin_table = pd.DataFrame(rows), pd.concat(bins, ignore_index=True)
    table.to_csv(OUT_TABLE, index=False)
    bin_table.to_csv(OUT_BINS, index=False)

    cols = ["split", "threshold", "coverage", "pct_abstained", "macro_f1_answered",
            "accuracy_answered", "pct_errors_caught", "pct_correct_lost"]
    print("=== TEST rows, locked threshold (DIAGNOSTIC ONLY) ===")
    print(table[cols].round(3).to_string(index=False))
    print("\n=== TEST: accuracy by confidence bucket ===")
    print(bin_table.round(3).to_string(index=False))
    print(f"\nSaved {OUT_TABLE.name}, {OUT_BINS.name}")


if __name__ == "__main__":
    main()