"""Step 2 pipeline: data/raw/*.jsonl -> data/processed/comments.csv

Dedups, drops too-short / too-long and non-Tunisian-looking text, normalizes, and tags
each surviving comment with its detected script. Spam/link-only rows are dropped.

Run:  python -m src.cleaning.clean
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import PROCESSED, RAW, ensure_data_dirs, settings
from .normalize import normalize
from .script_detect import detect_script

OUT = PROCESSED / "comments.csv"


def _load_raw() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(RAW.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def clean(out: Path = OUT) -> pd.DataFrame:
    ensure_data_dirs()
    cfg = settings()
    cc = cfg["cleaning"]
    sc = cfg["scraping"]

    rows = _load_raw()
    if not rows:
        print(f"No raw data in {RAW}. Run the scrapers first.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["clean_text"] = df["text"].astype(str).map(
        lambda t: normalize(
            t,
            strip_diacritics_=cc["strip_diacritics"],
            strip_tatweel_=cc["strip_tatweel"],
            normalize_letters_=cc["normalize_letters"],
            collapse_repeats_=cc["collapse_repeats"],
            max_repeat_run=cc["max_repeat_run"],
        )
    )

    df["script"] = df["clean_text"].map(detect_script)

    before = len(df)
    # length filter
    lengths = df["clean_text"].str.len()
    df = df[(lengths >= sc["min_chars"]) & (lengths <= sc["max_chars"])]
    # drop rows with no alphabetic content (emoji/number/punct-only spam)
    df = df[df["script"] != "other"]
    # dedup on normalized text
    if cc["dedup"]:
        df = df.drop_duplicates(subset="clean_text")

    df = df.reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")

    print(f"Cleaned {before} -> {len(df)} comments  ({before - len(df)} dropped)")
    print("By script:\n" + df["script"].value_counts().to_string())
    print(f"Wrote -> {out}")
    return df


if __name__ == "__main__":
    clean()
