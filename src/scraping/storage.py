"""Shared JSONL writer for scrapers. Deduplicates by id within a file."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

COMMENT_FIELDS = ("id", "platform", "text", "source", "timestamp")


def make_comment(id: str, platform: str, text: str, source: str,
                 timestamp: str | None = None) -> dict:
    return {"id": id, "platform": platform, "text": text,
            "source": source, "timestamp": timestamp}


def write_jsonl(path: Path, rows: Iterable[dict], append: bool = True) -> int:
    """Write rows as JSONL, skipping ids already present. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if append and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    written = 0
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written
