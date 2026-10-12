"""Tests for the step-3 labeling tooling (sampler + kappa)."""
import pandas as pd
import pytest

from src.labeling.sample import stratified_take


def _df():
    rows = (
        [{"id": f"a{i}", "script": "arabic", "clean_text": "x"} for i in range(100)]
        + [{"id": f"l{i}", "script": "latin", "clean_text": "y"} for i in range(300)]
        + [{"id": f"m{i}", "script": "mixed", "clean_text": "z"} for i in range(20)]
    )
    return pd.DataFrame(rows)


def test_stratified_take_exact_size():
    out = stratified_take(_df(), 84, "script", seed=42)
    assert len(out) == 84


def test_stratified_take_is_proportional():
    out = stratified_take(_df(), 84, "script", seed=42)
    vc = out["script"].value_counts()
    # latin is 300/420 ~ 71% -> ~60 of 84; arabic ~24%; mixed ~5%
    assert vc["latin"] > vc["arabic"] > vc["mixed"]
    assert vc.sum() == 84


def test_stratified_take_caps_at_availability():
    # ask for more than exists -> return everything, no duplication
    df = _df()
    out = stratified_take(df, 10_000, "script", seed=1)
    assert len(out) == len(df)
    assert out["id"].is_unique


def test_stratified_take_reproducible():
    a = stratified_take(_df(), 50, "script", seed=7)
    b = stratified_take(_df(), 50, "script", seed=7)
    assert sorted(a["id"]) == sorted(b["id"])


def test_kappa_compare(tmp_path):
    pytest.importorskip("sklearn")
    from src.labeling.kappa import compare

    a = pd.DataFrame({"id": ["1", "2", "3", "4"],
                      "label": ["positive", "negative", "positive", "neutral"]})
    b = pd.DataFrame({"id": ["1", "2", "3", "4"],
                      "label": ["positive", "negative", "negative", "neutral"]})
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    a.to_csv(pa, index=False)
    b.to_csv(pb, index=False)

    res = compare(pa, pb)
    assert res["n_overlap"] == 4
    assert res["raw_agreement"] == 0.75
    assert res["n_posneg_both"] == 3   # id4 is neutral for both -> excluded
    assert -1.0 <= res["cohen_kappa_3way"] <= 1.0
