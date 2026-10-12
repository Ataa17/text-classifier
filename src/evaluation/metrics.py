"""Evaluation metrics. Primary metric is macro-F1 (not accuracy).

Also provides a by-script breakdown (step 5 error analysis) and Cohen's kappa (step 3).
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    f1_score,
)


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    return f1_score(y_true, y_pred, average="macro")


def full_report(y_true: Sequence, y_pred: Sequence) -> str:
    return classification_report(y_true, y_pred, digits=3)


def kappa(labels_a: Sequence, labels_b: Sequence) -> float:
    """Cohen's kappa for the double-labeled overlap set (inter-annotator agreement)."""
    return cohen_kappa_score(labels_a, labels_b)


def by_script(y_true: Sequence, y_pred: Sequence, scripts: Sequence) -> pd.DataFrame:
    """Macro-F1 and support per detected script (arabic / latin / mixed)."""
    df = pd.DataFrame({"true": y_true, "pred": y_pred, "script": scripts})
    out = (
        df.groupby("script")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "macro_f1": macro_f1(g["true"], g["pred"]),
        }), include_groups=False)
        .reset_index()
    )
    return out
