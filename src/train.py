"""Phase 4: TF-IDF + logistic regression baseline, trained and validated per split.

Uses only each split's TRAIN rows (train part + validation part). Test rows are
never read here.

Usage:
    python -m src.train --model tfidf
    python -m src.train --model tfidf --split split_a
    python -m src.train --model tfidf --seed 7
"""
import argparse
import json
import sys
import time

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import (
    DATA_PATH, LABELS, LR_C_GRID, LR_CLASS_WEIGHT_GRID, LR_MAX_ITER, MODELS_DIR,
    RANDOM_SEED, RESULTS_DIR, SPLIT_COLUMNS, TARGET_COL, TFIDF_PARAMS,
    VALIDATION_FRACTION, ensure_dirs, set_global_seed,
)
from src.data import assert_no_leakage, load_raw, make_split
from src.evaluate import compute_metrics


def make_classifier(C, class_weight, seed) -> LogisticRegression:
    # Default lbfgs solver with >2 classes is multinomial (softmax) in sklearn 1.5.
    return LogisticRegression(
        C=C, class_weight=class_weight, max_iter=LR_MAX_ITER, random_state=seed
    )


def run_tfidf(df, split_col, seed) -> None:
    sd = make_split(df, split_col, seed=seed)
    assert_no_leakage(df, sd)
    print(f"\n=== {split_col}: train={len(sd.train)} val={len(sd.val)} "
          f"(test={len(sd.test)} rows NOT used) ===")

    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(sd.train["text"])  # vocabulary + IDF from TRAIN only
    X_val = vectorizer.transform(sd.val["text"])
    y_train, y_val = sd.train[TARGET_COL], sd.val[TARGET_COL]
    print(f"vocabulary size: {len(vectorizer.vocabulary_)}")

    grid_rows, best = [], None
    for class_weight in LR_CLASS_WEIGHT_GRID:
        for C in LR_C_GRID:
            start = time.time()
            clf = make_classifier(C, class_weight, seed).fit(X_train, y_train)
            metrics = compute_metrics(y_val, clf.predict(X_val), LABELS)
            grid_rows.append({
                "class_weight": str(class_weight),
                "C": C,
                "val_macro_f1": round(metrics["macro_f1"], 4),
                "val_accuracy": round(metrics["accuracy"], 4),
                "fit_seconds": round(time.time() - start, 1),
            })
            if best is None or metrics["macro_f1"] > best["metrics"]["macro_f1"]:
                best = {"clf": clf, "C": C, "class_weight": class_weight, "metrics": metrics}

    grid = pd.DataFrame(grid_rows)
    print("\nValidation grid (selection metric: macro-F1):")
    print(grid.to_string(index=False))
    print(f"\nSelected: C={best['C']}, class_weight={best['class_weight']}  "
          f"-> val macro-F1={best['metrics']['macro_f1']:.4f}, "
          f"val accuracy={best['metrics']['accuracy']:.4f}")
    print("\nPer-class validation metrics for the selected model:")
    print(pd.DataFrame(best["metrics"]["per_class"]).T.round(3).to_string())

    # --- Save everything ------------------------------------------------------
    clf = best["clf"]
    classes = [str(c) for c in clf.classes_]  # column order of predict_proba
    proba = clf.predict_proba(X_val)
    preds = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes])
    preds.insert(0, "confidence", proba.max(axis=1))
    preds.insert(0, "pred", [classes[i] for i in proba.argmax(axis=1)])
    preds.insert(0, "true", y_val.values)
    preds.insert(0, "id", sd.val["id"].values)

    tag = f"tfidf_{split_col}"
    model_dir = MODELS_DIR / tag
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    joblib.dump(clf, model_dir / "classifier.joblib")
    (model_dir / "labels.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    config = {
        "model": "tfidf_logreg",
        "split": split_col,
        "seed": seed,
        "validation_fraction": VALIDATION_FRACTION,
        "tfidf_params": TFIDF_PARAMS,
        "selected": {"C": best["C"], "class_weight": best["class_weight"]},
        "grid": {"C": list(LR_C_GRID), "class_weight": [str(c) for c in LR_CLASS_WEIGHT_GRID]},
        "selection_metric": "validation macro-F1",
        "n_train": len(sd.train),
        "n_val": len(sd.val),
        "sklearn_version": sklearn.__version__,
        "fitted_on": "train rows minus validation rows; refit on full train happens in Phase 6",
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    grid.to_csv(RESULTS_DIR / f"{tag}_val_grid.csv", index=False)
    (RESULTS_DIR / f"{tag}_val_metrics.json").write_text(
        json.dumps(best["metrics"], indent=2), encoding="utf-8")
    preds.to_csv(RESULTS_DIR / f"{tag}_val_predictions.csv", index=False)
    print(f"\nSaved model to {model_dir} and validation files to results/{tag}_val_*")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tfidf"], default="tfidf")
    ap.add_argument("--split", choices=[*SPLIT_COLUMNS, "all"], default="all")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--data", default=str(DATA_PATH))
    args = ap.parse_args()

    ensure_dirs()
    set_global_seed(args.seed)
    df = load_raw(args.data)
    splits = SPLIT_COLUMNS if args.split == "all" else (args.split,)
    for split_col in splits:
        run_tfidf(df, split_col, args.seed)


if __name__ == "__main__":
    main()