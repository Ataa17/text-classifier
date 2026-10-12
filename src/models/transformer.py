"""Step 4 transformers — fine-tune and compare dialect-aware encoders.

Candidates (compare macro-F1):
    - DziriBERT      (alioui/dziribert)          Algerian dialect, close to Tunisian
    - MARBERT        (UBC-NLP/MARBERT)           dialectal Arabic
    - CAMeLBERT-DA   (CAMeL-Lab/bert-base-arabic-camelbert-da)

STUB: fill in with a HuggingFace Trainer in step 4. Use fixed seeds and the same
stratified split as the baseline so numbers are comparable.
"""
from __future__ import annotations

MODELS = {
    "dziribert": "alioui/dziribert",
    "marbert": "UBC-NLP/MARBERT",
    "camelbert-da": "CAMeL-Lab/bert-base-arabic-camelbert-da",
}


def finetune(model_key: str, *args, **kwargs):
    raise NotImplementedError("Implemented in step 4.")
