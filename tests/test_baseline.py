"""Tests for the step-4 baseline (data loader + training on synthetic data)."""
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from src.models.data import load_labeled  # noqa: E402


def test_load_labeled_filters_to_posneg(tmp_path):
    csv = tmp_path / "labeled.csv"
    pd.DataFrame({
        "id": ["1", "2", "3", "4", "5"],
        "clean_text": ["a", "b", "c", "d", "e"],
        "script": ["latin"] * 5,
        "label": ["positive", "negative", "neutral", "", "positive"],
    }).to_csv(csv, index=False)

    df = load_labeled(paths=[csv], include_kappa=False)
    assert set(df["label"]) == {"positive", "negative"}
    assert len(df) == 3  # neutral + blank dropped


def _synthetic(n_per_class=40):
    pos = [f"good great love excellent {i}" for i in range(n_per_class)]
    neg = [f"bad awful hate terrible {i}" for i in range(n_per_class)]
    rows = (
        [{"id": f"p{i}", "clean_text": t, "script": "latin", "label": "positive"}
         for i, t in enumerate(pos)]
        + [{"id": f"n{i}", "clean_text": t, "script": "latin", "label": "negative"}
           for i, t in enumerate(neg)]
    )
    return pd.DataFrame(rows)


def test_baseline_trains_and_reports(tmp_path, monkeypatch):
    from src.models import baseline

    # redirect artifacts into tmp so the test doesn't touch the repo
    monkeypatch.setattr(baseline, "MODELS", tmp_path / "models")
    monkeypatch.setattr(baseline, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(baseline, "MODEL_PATH", tmp_path / "models" / "baseline.joblib")

    metrics = baseline.run(df=_synthetic(), save=True)

    assert 0.0 <= metrics["test_macro_f1"] <= 1.0
    # separable synthetic data -> the baseline should do clearly better than chance
    assert metrics["test_macro_f1"] > 0.8
    assert (tmp_path / "models" / "baseline.joblib").exists()
    assert (tmp_path / "reports" / "baseline_metrics.json").exists()
