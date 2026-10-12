"""Step 3 — LLM pre-labeling of the MAIN set, to speed up human review.

Writes a `suggestion` column (positive / negative / neutral) into to_label_main.csv.
The human then confirms or corrects every suggestion in the labeling CLI (Enter = accept).

NEVER run this on the kappa overlap set — that must be labeled blind for an honest kappa.

Uses OpenAI (ChatGPT) via OPENAI_API_KEY (from .env). Resumable: only fills rows with no
suggestion yet, saves after every batch.

Run:  python -m src.labeling.llm_prelabel
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config import LABELED, env, settings

OUT_MAIN = LABELED / "to_label_main.csv"
VALID = {"positive", "negative", "neutral"}

SYSTEM = (
    "You label the sentiment of Tunisian Arabic (Derja) social-media comments. Derja mixes "
    "Arabic script, Latin-script 'Arabizi' (e.g. 'behi barcha'), and French.\n"
    "Label each comment as exactly one of:\n"
    "  positive  - approval, praise, affection, joy, support, gratitude, satisfaction\n"
    "  negative  - criticism, anger, disgust, mockery, disappointment, insult, sadness\n"
    "  neutral   - factual/informational, a plain question, greeting/tagging only, spam, "
    "off-topic, unintelligible, or sentiment that is genuinely balanced\n"
    "Judge the INTENDED meaning: sarcasm/irony flips the literal words. Emojis count. "
    "Negation matters ('mech behi' = not good = negative). The target of the sentiment "
    "does not matter. These are only first-pass guesses a human will verify.\n"
    "Respond with ONLY a JSON array, one object per comment in order: "
    '[{"i": 0, "label": "positive"}, ...]. No prose, no code fences.'
)


def _parse_labels(raw: str, n: int) -> list[str]:
    """Extract n labels from the model's JSON reply; unknown/missing -> '' (left for human)."""
    out = [""] * n
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return out
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return out
    for obj in data:
        try:
            i, label = int(obj["i"]), str(obj["label"]).strip().lower()
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= i < n and label in VALID:
            out[i] = label
    return out


def _classify_batch(client, texts: list[str], model: str, max_tokens: int) -> list[str]:
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(texts))
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": numbered},
        ],
    )
    text = resp.choices[0].message.content or ""
    return _parse_labels(text, len(texts))


def prelabel(path: Path = OUT_MAIN) -> None:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python -m src.labeling.sample` first.")
    env("OPENAI_API_KEY", required=True)  # fail early with a clear message

    from openai import OpenAI  # imported here so the rest of the project doesn't require the SDK

    lc = settings()["labeling"]["llm"]
    df = pd.read_csv(path, dtype={"suggestion": "string"})
    if "suggestion" not in df.columns:
        df["suggestion"] = ""
    df["suggestion"] = df["suggestion"].fillna("")

    todo = df.index[df["suggestion"] == ""].tolist()
    if not todo:
        print("All rows already have a suggestion. Nothing to do.")
        return
    print(f"Pre-labeling {len(todo)} comments with {lc['model']} ...")

    client = OpenAI()
    bs = lc["batch_size"]
    for start in tqdm(range(0, len(todo), bs)):
        idxs = todo[start:start + bs]
        texts = df.loc[idxs, "clean_text"].astype(str).tolist()
        try:
            labels = _classify_batch(client, texts, lc["model"], lc["max_tokens"])
        except Exception as e:  # keep going; unlabeled rows just stay for manual review
            print(f"  batch at {start} failed ({e}); leaving blank")
            continue
        for idx, label in zip(idxs, labels):
            df.loc[idx, "suggestion"] = label
        df.to_csv(path, index=False, encoding="utf-8")

    got = int((df["suggestion"] != "").sum())
    print(f"Done. {got}/{len(df)} rows have a suggestion.")
    print(df["suggestion"].value_counts().to_string())


if __name__ == "__main__":
    prelabel()
