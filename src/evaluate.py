"""Metric helpers (Phases 4-5) and the final test evaluation (Phase 6).

Usage:
    python -m src.evaluate select   # choose the model from VALIDATION results only
    python -m src.evaluate test     # refit on full train, score the test rows once
"""
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import (
    DATA_PATH, EMBED_MODEL_NAME, LABELS, MODEL_NAMES, MODELS_DIR, RANDOM_SEED,
    RESULTS_DIR, SPLIT_COLUMNS, TARGET_COL, TFIDF_PARAMS, ensure_dirs, set_global_seed,
)
from src.data import assert_no_leakage, load_raw, make_split

SELECTION_PATH = RESULTS_DIR / "model_selection.json"


# --- Metrics ------------------------------------------------------------------

def compute_metrics(y_true, y_pred, labels) -> dict:
    """Macro-F1, accuracy, per-class precision/recall/F1, confusion matrix.

    Rows of the confusion matrix = true label, columns = predicted label,
    both in the order of `labels`.
    """
    labels = list(labels)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            lab: {
                "precision": float(report[lab]["precision"]),
                "recall": float(report[lab]["recall"]),
                "f1": float(report[lab]["f1-score"]),
                "support": int(report[lab]["support"]),
            }
            for lab in labels
        },
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def report_table(metrics: dict) -> pd.DataFrame:
    """Per-class precision/recall/F1/support plus a macro-average row."""
    t = pd.DataFrame(metrics["per_class"]).T[["precision", "recall", "f1", "support"]]
    t.loc["macro avg"] = [t["precision"].mean(), t["recall"].mean(),
                          t["f1"].mean(), t["support"].sum()]
    return t.round(4)


def plot_confusion(metrics: dict, title: str, path) -> None:
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    labels = metrics["labels"]
    cm = np.array(metrics["confusion_matrix"])
    row_pct = cm / cm.sum(axis=1, keepdims=True)  # share of each TRUE class

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{row_pct[i, j]:.0%}", ha="center", va="center",
                    fontsize=7, color="white" if row_pct[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="share of the true class (row-normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Step 1: model selection from VALIDATION results only -----------------------

def select_models() -> None:
    rows = []
    for split_col in SPLIT_COLUMNS:
        for model in MODEL_NAMES:
            path = RESULTS_DIR / f"{model}_{split_col}_val_metrics.json"
            m = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"split": split_col, "model": model,
                         "val_macro_f1": round(m["macro_f1"], 4),
                         "val_accuracy": round(m["accuracy"], 4)})
    table = pd.DataFrame(rows)
    best_per_split = {
        split: str(g.loc[g["val_macro_f1"].idxmax(), "model"])
        for split, g in table.groupby("split")
    }
    mean_f1 = table.groupby("model")["val_macro_f1"].mean()
    common = str(mean_f1.idxmax())

    selection = {
        "rule": "highest validation macro-F1; the common model is the one with the "
                "higher mean validation macro-F1 across split_a and split_b",
        "validation": rows,
        "best_per_split": best_per_split,
        "mean_val_macro_f1": {k: round(float(v), 4) for k, v in mean_f1.items()},
        "common_model": common,
    }
    SELECTION_PATH.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nBest per split (validation): {best_per_split}")
    print(f"Mean validation macro-F1   : {selection['mean_val_macro_f1']}")
    print(f"Common model chosen        : {common}")
    print(f"\nSaved {SELECTION_PATH.name}. COMMIT IT before running the test step.")


# --- Step 2: refit on full train, score the test rows once ----------------------

def _load_locked(model: str, split_col: str) -> dict:
    cfg = json.loads((MODELS_DIR / f"{model}_{split_col}" / "config.json").read_text(encoding="utf-8"))
    if model == "tfidf":
        if json.loads(json.dumps(TFIDF_PARAMS)) != cfg["tfidf_params"]:
            raise RuntimeError("TFIDF_PARAMS changed since validation; re-run src.train first.")
    elif cfg["encoder"] != EMBED_MODEL_NAME:
        raise RuntimeError("EMBED_MODEL_NAME changed since validation; re-run src.train first.")
    return cfg


def fit_final(model, split_col, df, encoder):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from src.train import make_classifier  # lazy import: train.py imports this module

    cfg = _load_locked(model, split_col)
    sd = make_split(df, split_col, seed=cfg["seed"], val_fraction=cfg["validation_fraction"])
    assert_no_leakage(df, sd)
    full = pd.concat([sd.train, sd.val], ignore_index=True)  # == all rows marked train
    print(f"\n=== {model} / {split_col}: fit on {len(full)} train rows, "
          f"scoring {len(sd.test)} test rows ===")

    if model == "tfidf":
        vec = TfidfVectorizer(**TFIDF_PARAMS)
        X_full = vec.fit_transform(full["text"])
        X_test = vec.transform(sd.test["text"])
        extra = {"vectorizer": vec}
    else:
        from src.embeddings import encode_texts
        X_full = encode_texts(encoder, full["text"])
        X_test = encode_texts(encoder, sd.test["text"])
        extra = {}

    sel = cfg["selected"]
    clf = make_classifier(sel["C"], sel["class_weight"], cfg["seed"]).fit(X_full, full[TARGET_COL])
    return sd, full, clf, extra, X_test, cfg


def evaluate_one(model, split_col, df, encoder, force) -> None:
    tag = f"test_{model}_{split_col}"
    metrics_path = RESULTS_DIR / f"{tag}_metrics.json"
    if metrics_path.exists() and not force:
        print(f"[skip] {tag}: results already exist. Test rows are scored once; "
              f"use --force only to recover from a crash.")
        return

    sd, full, clf, extra, X_test, cfg = fit_final(model, split_col, df, encoder)
    classes = [str(c) for c in clf.classes_]
    proba = clf.predict_proba(X_test)
    pred = [classes[i] for i in proba.argmax(axis=1)]
    y_test = sd.test[TARGET_COL].values

    metrics = compute_metrics(y_test, pred, LABELS)
    metrics.update({"model": model, "split": split_col, "n_fit": len(full),
                    "n_test": len(sd.test), "locked_params": cfg["selected"]})
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = report_table(metrics)
    report.to_csv(RESULTS_DIR / f"{tag}_classification_report.csv")
    plot_confusion(metrics, f"{model} on {split_col} test "
                            f"(macro-F1 {metrics['macro_f1']:.3f})",
                   RESULTS_DIR / f"{tag}_confusion_matrix.png")

    preds = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes])
    preds.insert(0, "confidence", proba.max(axis=1))
    preds.insert(0, "pred", pred)
    preds.insert(0, "true", y_test)
    preds.insert(0, "id", sd.test["id"].values)
    preds.to_csv(RESULTS_DIR / f"{tag}_predictions.csv", index=False)

    out = MODELS_DIR / "final" / f"{model}_{split_col}"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "classifier.joblib")
    for name, obj in extra.items():
        joblib.dump(obj, out / f"{name}.joblib")
    (out / "labels.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    final_cfg = {**cfg, "fitted_on": "all rows marked train (train + validation parts)",
                 "n_fit": len(full)}
    (out / "config.json").write_text(json.dumps(final_cfg, indent=2), encoding="utf-8")

    print(f"test macro-F1 = {metrics['macro_f1']:.4f}   accuracy = {metrics['accuracy']:.4f}")
    print(report.to_string())


def build_summaries() -> None:
    sel = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    val_f1 = {(r["split"], r["model"]): r["val_macro_f1"] for r in sel["validation"]}
    summary, per_class = [], []
    for split_col in SPLIT_COLUMNS:
        for model in MODEL_NAMES:
            path = RESULTS_DIR / f"test_{model}_{split_col}_metrics.json"
            if not path.exists():
                continue
            m = json.loads(path.read_text(encoding="utf-8"))
            summary.append({
                "split": split_col, "model": model,
                "val_macro_f1": val_f1[(split_col, model)],
                "test_macro_f1": round(m["macro_f1"], 4),
                "test_accuracy": round(m["accuracy"], 4),
                "n_test": m["n_test"],
                "chosen_on_validation": model == sel["common_model"],
            })
            for lab, v in m["per_class"].items():
                per_class.append({"split": split_col, "model": model, "class": lab,
                                  "precision": round(v["precision"], 4),
                                  "recall": round(v["recall"], 4),
                                  "f1": round(v["f1"], 4), "support": v["support"]})
    summary_df, per_class_df = pd.DataFrame(summary), pd.DataFrame(per_class)
    summary_df.to_csv(RESULTS_DIR / "test_macro_f1_summary.csv", index=False)
    per_class_df.to_csv(RESULTS_DIR / "test_per_class_metrics.csv", index=False)
    print("\n=== Macro-F1 summary (validation vs final test) ===")
    print(summary_df.to_string(index=False))
    print("\n=== Test F1 per class ===")
    print(per_class_df.pivot(index="class", columns=["split", "model"], values="f1")
          .reindex(list(LABELS)).to_string())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("select", help="choose the model from validation results only")
    t = sub.add_parser("test", help="refit on full train and score the test rows once")
    t.add_argument("--force", action="store_true", help="recover from a crash only")
    t.add_argument("--data", default=str(DATA_PATH))
    args = ap.parse_args()

    ensure_dirs()
    if args.command == "select":
        select_models()
        return

    if not SELECTION_PATH.exists():
        raise SystemExit("Run `python -m src.evaluate select` first, and commit its output.")
    set_global_seed(RANDOM_SEED)
    df = load_raw(args.data)
    from src.embeddings import load_encoder
    encoder = load_encoder()
    print("NOTE: test rows are read from here on.")
    for split_col in SPLIT_COLUMNS:
        for model in MODEL_NAMES:
            evaluate_one(model, split_col, df, encoder, args.force)
    build_summaries()


if __name__ == "__main__":
    main()