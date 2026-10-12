"""Step 3 — terminal hand-labeling tool. Resumable; writes after every label.

Serves unlabeled rows one at a time from a to_label CSV and records your decision.

Keys:   p = positive    n = negative    d = discard (neutral / not applicable)
        s = skip (decide later)    u = undo last    q = quit & save

Usage:  python -m src.labeling.label_cli            # labels to_label_main.csv
        python -m src.labeling.label_cli kappa      # labels to_label_kappa.csv
        python -m src.labeling.label_cli <path.csv>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from ..config import LABELED

# Make Arabic render on a cp1252 Windows console.
for _stream in (sys.stdout, sys.stdin):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

KEYS = {"p": "positive", "n": "negative", "d": "neutral"}


def _resolve_target(arg: str | None) -> Path:
    if arg is None or arg == "main":
        return LABELED / "to_label_main.csv"
    if arg == "kappa":
        return LABELED / "to_label_kappa.csv"
    return Path(arg)


def label(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python -m src.labeling.sample` first.")

    df = pd.read_csv(path, dtype={"label": "string"})
    df["label"] = df["label"].fillna("")

    todo = df.index[df["label"] == ""].tolist()
    done = int((df["label"] != "").sum())
    total = len(df)
    print(f"\n{path.name}: {done}/{total} labeled, {len(todo)} to go.")
    print("keys: [p]ositive  [n]egative  [d]iscard  [s]kip  [u]ndo  [q]uit\n")

    history: list[int] = []
    i = 0
    while i < len(todo):
        idx = todo[i]
        row = df.loc[idx]
        print("-" * 60)
        print(f"[{done + i + 1}/{total}]  ({row['script']} | {row['source']})")
        print(row["clean_text"])
        choice = input("> ").strip().lower()

        if choice == "q":
            break
        if choice == "s":
            i += 1
            continue
        if choice == "u":
            if history:
                prev = history.pop()
                df.loc[prev, "label"] = ""
                df.to_csv(path, index=False, encoding="utf-8")
                i -= 1
                # if the undone row was skipped it may not be the immediate previous; find it
                while i > 0 and todo[i] != prev:
                    i -= 1
            else:
                print("(nothing to undo)")
            continue
        if choice in KEYS:
            df.loc[idx, "label"] = KEYS[choice]
            df.to_csv(path, index=False, encoding="utf-8")
            history.append(idx)
            i += 1
        else:
            print("(?) use p / n / d / s / u / q")

    labeled = df[df["label"].isin(["positive", "negative"])]
    discarded = int((df["label"] == "neutral").sum())
    print("\nSaved.")
    print(f"  pos/neg kept : {len(labeled)}  "
          f"({(labeled['label'] == 'positive').sum()} pos / "
          f"{(labeled['label'] == 'negative').sum()} neg)")
    print(f"  discarded    : {discarded}")
    print(f"  still to do  : {int((df['label'] == '').sum())}")


if __name__ == "__main__":
    label(_resolve_target(sys.argv[1] if len(sys.argv) > 1 else None))
