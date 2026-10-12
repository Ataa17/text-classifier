"""Step 3 — draw the labeling sets from data/processed/comments.csv.

Produces two disjoint, script-stratified CSVs in data/labeled/ with an empty `label` column:
  - to_label_kappa.csv : the blind double-labeling set (Cohen's kappa)
  - to_label_main.csv  : the bulk hand-labeling set

Reproducible (fixed seed). Refuses to overwrite existing files so in-progress labels are safe.

Run:  python -m src.labeling.sample
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ..config import LABELED, PROCESSED, ensure_data_dirs, settings

SRC = PROCESSED / "comments.csv"
OUT_MAIN = LABELED / "to_label_main.csv"
OUT_KAPPA = LABELED / "to_label_kappa.csv"
COLUMNS = ["id", "source", "script", "clean_text", "label"]


def stratified_take(df: pd.DataFrame, n: int, col: str, seed: int) -> pd.DataFrame:
    """Take ~n rows from df, allocated across `col` groups by the largest-remainder
    method and capped at each group's availability. Returns the sampled rows."""
    n = min(n, len(df))
    counts = df[col].value_counts()
    total = len(df)

    raw = {g: counts[g] / total * n for g in counts.index}
    alloc = {g: min(int(math.floor(r)), counts[g]) for g, r in raw.items()}

    # distribute the remainder by largest fractional part, respecting caps
    remaining = n - sum(alloc.values())
    frac_order = sorted(raw, key=lambda g: raw[g] - math.floor(raw[g]), reverse=True)
    i = 0
    while remaining > 0 and frac_order:
        g = frac_order[i % len(frac_order)]
        if alloc[g] < counts[g]:
            alloc[g] += 1
            remaining -= 1
        i += 1
        if i > 10000:  # safety against pathological loops
            break

    parts = [
        df[df[col] == g].sample(n=k, random_state=seed)
        for g, k in alloc.items() if k > 0
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed)  # shuffle


def sample(force: bool = False) -> None:
    ensure_data_dirs()
    if not force and (OUT_MAIN.exists() or OUT_KAPPA.exists()):
        raise SystemExit(
            f"Refusing to overwrite existing labeling files in {LABELED}. "
            "Delete them or pass force=True if you really mean to re-sample."
        )
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found. Run `python -m src.cleaning.clean` first.")

    cfg = settings()
    seed = cfg["project"]["seed"]
    lc = cfg["labeling"]
    strat_col = "script" if lc["stratify_by_script"] else None

    df = pd.read_csv(SRC)
    df = df.drop_duplicates(subset="id").reset_index(drop=True)

    def take(frame, n):
        if strat_col:
            return stratified_take(frame, n, strat_col, seed)
        return frame.sample(n=min(n, len(frame)), random_state=seed)

    kappa = take(df, lc["kappa_overlap"])
    rest = df[~df["id"].isin(kappa["id"])]
    main = take(rest, lc["sample_size"])

    for frame, out in ((kappa, OUT_KAPPA), (main, OUT_MAIN)):
        frame = frame.copy()
        frame["label"] = ""
        frame[COLUMNS].to_csv(out, index=False, encoding="utf-8")
        print(f"Wrote {len(frame):>5} rows -> {out.name}")
        print("   by script: " + frame["script"].value_counts().to_dict().__str__())


if __name__ == "__main__":
    sample()
