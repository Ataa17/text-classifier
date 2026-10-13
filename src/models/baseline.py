"""Step 4 baseline — TF-IDF char n-grams + Logistic Regression.

Char n-grams are robust to the heavy misspelling / spelling variation in Arabizi.
Trains on the hand-labeled pos/neg data, reports macro-F1 + per-class + by-script on a
stratified held-out test set, and saves the fitted model.

Run (once labeled data exists):  python -m src.models.baseline
"""
from __future__ import annotations

import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ..config import ROOT, settings
from ..evaluation.metrics import by_script, full_report, macro_f1
from .data import load_labeled

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODEL_PATH = MODELS / "baseline.joblib"


def build_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5),
            min_df=2, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed,
        )),
    ])


def run(df=None, save: bool = True) -> dict:
    cfg = settings()
    seed = cfg["project"]["seed"]
    test_size = cfg["split"]["test_size"]
    stratify_cfg = cfg["split"]["stratify"]

    if df is None:
        df = load_labeled()

    X = df["clean_text"].astype(str)
    y = df["label"].astype(str)
    scripts = df["script"].astype(str)

    X_tr, X_te, y_tr, y_te, _, scr_te = train_test_split(
        X, y, scripts,
        test_size=test_size,
        stratify=y if stratify_cfg else None,
        random_state=seed,
    )

    pipe = build_pipeline(seed)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    dist = y.value_counts().to_dict()
    metrics = {
        "n_total": len(df),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "class_distribution": {k: int(v) for k, v in dist.items()},
        "test_macro_f1": round(float(macro_f1(y_te, pred)), 4),
    }

    report = full_report(y_te, pred)
    by_scr = by_script(y_te, pred, scr_te)
    print(f"train={metrics['n_train']}  test={metrics['n_test']}  "
          f"classes={metrics['class_distribution']}")
    print(f"\nTEST macro-F1: {metrics['test_macro_f1']}\n")
    print(report)
    print("\nBy script:\n" + by_scr.to_string(index=False))

    if save:
        MODELS.mkdir(parents=True, exist_ok=True)
        REPORTS.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)
        (REPORTS / "baseline_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8")
        (REPORTS / "baseline_report.txt").write_text(
            report + "\n\nBy script:\n" + by_scr.to_string(index=False), encoding="utf-8")
        print(f"\nSaved model -> {MODEL_PATH}")
        print(f"Saved metrics -> {REPORTS / 'baseline_metrics.json'}")

    return metrics


if __name__ == "__main__":
    run()
