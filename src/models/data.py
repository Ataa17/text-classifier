"""Load hand-labeled data for training (step 4).

Reads the labeled CSVs, keeps only confirmed pos/neg rows (drops neutral/discarded and
anything still unlabeled), dedups by id. The `suggestion` column (LLM guesses) is ignored —
only the human `label` is ground truth.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import LABELED

MAIN = LABELED / "to_label_main.csv"
KAPPA = LABELED / "to_label_kappa.csv"
CLASSES = ["negative", "positive"]


def load_labeled(paths: list[Path] | None = None, include_kappa: bool = True) -> pd.DataFrame:
    """Return a frame with columns [id, clean_text, script, label], label in {pos, neg}."""
    if paths is None:
        paths = [MAIN] + ([KAPPA] if include_kappa else [])

    frames = []
    for p in paths:
        p = Path(p)
        if p.exists():
            frames.append(pd.read_csv(p, dtype={"label": "string"}))
    if not frames:
        raise SystemExit(
            "No labeled data found. Label some comments first "
            "(python -m src.labeling.label_cli)."
        )

    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].fillna("")
    df = df[df["label"].isin(CLASSES)].drop_duplicates(subset="id")
    if df.empty:
        raise SystemExit("No pos/neg labels yet — nothing to train on.")
    return df[["id", "clean_text", "script", "label"]].reset_index(drop=True)
