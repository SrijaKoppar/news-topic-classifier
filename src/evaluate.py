"""Metric helpers shared by training (validation) and final evaluation."""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


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