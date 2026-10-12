"""Step 3 — Cohen's kappa for the blind double-labeled overlap set.

Give it the two annotators' completed copies of to_label_kappa.csv. It aligns on comment id,
scores agreement over the three decisions {positive, negative, neutral}, and also reports
agreement restricted to the pos/neg comments (the actual task).

Usage:  python -m src.labeling.kappa annotator_a.csv annotator_b.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from ..evaluation.metrics import kappa as cohen_kappa


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"label": "string", "id": "string"})
    df["label"] = df["label"].fillna("")
    return df[df["label"] != ""][["id", "label"]]


def compare(path_a: Path, path_b: Path) -> dict:
    a, b = _load(path_a), _load(path_b)
    merged = a.merge(b, on="id", suffixes=("_a", "_b"))
    if merged.empty:
        raise SystemExit("No overlapping labeled ids between the two files.")

    n = len(merged)
    agree = int((merged["label_a"] == merged["label_b"]).sum())
    k3 = cohen_kappa(merged["label_a"], merged["label_b"])

    # restricted to items both annotators kept as pos/neg
    pn = merged[merged["label_a"].isin(["positive", "negative"])
                & merged["label_b"].isin(["positive", "negative"])]
    k2 = cohen_kappa(pn["label_a"], pn["label_b"]) if len(pn) > 1 else float("nan")

    result = {
        "n_overlap": n,
        "raw_agreement": agree / n,
        "cohen_kappa_3way": k3,
        "n_posneg_both": len(pn),
        "cohen_kappa_posneg": k2,
    }
    for key, val in result.items():
        print(f"  {key:22}: {val}")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m src.labeling.kappa <annotator_a.csv> <annotator_b.csv>")
    compare(Path(sys.argv[1]), Path(sys.argv[2]))
