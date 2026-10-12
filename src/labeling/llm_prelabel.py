"""Step 3 — optional LLM pre-labeling to speed up hand-correction.

STUB. Important: the Cohen's-kappa overlap set must be labeled BLIND (no LLM suggestion)
so inter-annotator agreement is honest. Only use pre-labels to accelerate the bulk set,
and always hand-correct.

Planned: read data/processed/comments.csv, prompt an LLM for pos/neg (with an explicit
"neutral -> discard" option), write data/labeled/prelabeled.csv for human review.
"""
from __future__ import annotations

LABELS = ("negative", "positive")  # neutral is discarded, not a class


def prelabel(*args, **kwargs):
    raise NotImplementedError("Implemented in step 3.")
