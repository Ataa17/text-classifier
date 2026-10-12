"""Central config: loads secrets from .env and settings from config/config.yaml."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project paths
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
LABELED = DATA / "labeled"

load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    """Parsed config/config.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def env(key: str, default: str | None = None, required: bool = False) -> str | None:
    """Read an environment variable (from .env or the process env)."""
    val = os.getenv(key, default)
    if required and not val:
        raise RuntimeError(
            f"Missing required env var {key!r}. Copy .env.example to .env and fill it in."
        )
    return val


def ensure_data_dirs() -> None:
    for d in (RAW, INTERIM, PROCESSED, LABELED):
        d.mkdir(parents=True, exist_ok=True)
