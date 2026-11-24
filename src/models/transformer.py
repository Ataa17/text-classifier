"""Step 4b — fine-tune and compare dialect-aware transformer encoders.

Candidates (compare test macro-F1 against the TF-IDF baseline):
    - DziriBERT      (alioui/dziribert)                       Algerian dialect, close to Tunisian
    - MARBERT        (UBC-NLP/MARBERT)                         dialectal Arabic
    - CAMeLBERT-DA   (CAMeL-Lab/bert-base-arabic-camelbert-da) dialectal Arabic

Heavy deps (torch/transformers/datasets) are imported lazily so the rest of the project —
and the test suite — don't require them. Fine-tuning is slow on CPU; use a GPU (the Colab
notebook in notebooks/finetune_colab.ipynb is the easy path).

Run (GPU recommended):  python -m src.models.transformer            # all models in config
                        python -m src.models.transformer marbert     # one model
"""
from __future__ import annotations

import json
import sys

from ..config import ROOT, settings
from .data import CLASSES, load_labeled

MODELS = {
    "tunbert": "tunis-ai/TunBERT",
    "dziribert": "alger-ia/dziribert",
    "marbert": "UBC-NLP/MARBERT",
    "camelbert-da": "CAMeL-Lab/bert-base-arabic-camelbert-da",
}

OUT_ROOT = ROOT / "models"
REPORTS = ROOT / "reports"
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}       # {"negative":0, "positive":1}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


def prepare_splits(df=None, test_size: float | None = None, seed: int | None = None):
    """Stratified train/test split. Pure (sklearn only) so it's testable without torch."""
    from sklearn.model_selection import train_test_split

    cfg = settings()
    seed = cfg["project"]["seed"] if seed is None else seed
    test_size = cfg["transformer"]["test_size"] if test_size is None else test_size
    if df is None:
        df = load_labeled()
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_metrics(eval_pred):
    """HF Trainer metric fn. Pure (numpy + sklearn), testable without torch."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    preds = np.asarray(logits).argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def finetune(model_key: str, df=None, epochs: int | None = None) -> dict:
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from ..evaluation.metrics import by_script, full_report, macro_f1

    if model_key not in MODELS:
        raise SystemExit(f"Unknown model '{model_key}'. Choose from: {', '.join(MODELS)}")
    tc = settings()["transformer"]
    hf_name = MODELS[model_key]
    trust_remote_code = model_key == "tunbert"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for transformer fine-tuning.")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    train_df, test_df = prepare_splits(df)
    tok = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=trust_remote_code)

    def to_ds(frame):
        ds = Dataset.from_dict({
            "text": frame["clean_text"].astype(str).tolist(),
            "label": [LABEL2ID[l] for l in frame["label"]],
        })
        return ds.map(
            lambda b: tok(b["text"], truncation=True, max_length=tc["max_length"]),
            batched=True,
        )

    train_ds, test_ds = to_ds(train_df), to_ds(test_df)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_name,
        num_labels=len(CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=trust_remote_code,
    )
    model.to("cuda")

    out_dir = OUT_ROOT / model_key
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs or tc["epochs"],
        per_device_train_batch_size=tc["batch_size"],
        per_device_eval_batch_size=tc["batch_size"],
        gradient_accumulation_steps=tc.get("gradient_accumulation_steps", 1),
        learning_rate=float(tc["learning_rate"]),
        fp16=torch.cuda.is_available(),
        save_strategy="no",
        logging_steps=50,
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics, processing_class=tok,
    )
    trainer.train()

    predictions = trainer.predict(test_ds).predictions
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    pred_ids = np.asarray(predictions).argmax(axis=-1)
    preds = [ID2LABEL[i] for i in pred_ids]

    metrics = {
        "model": model_key,
        "hf_name": hf_name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "test_macro_f1": round(float(macro_f1(test_df["label"], preds)), 4),
    }
    print(f"\n=== {model_key} ({hf_name}) ===")
    print(f"TEST macro-F1: {metrics['test_macro_f1']}\n")
    print(full_report(test_df["label"], preds))
    print("\nBy script:\n" + by_script(test_df["label"], preds, test_df["script"]).to_string(index=False))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (REPORTS / f"{model_key}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved model -> {out_dir}")
    return metrics


def run_all(df=None) -> list[dict]:
    if df is None:
        df = load_labeled()
    results = [finetune(k, df=df) for k in settings()["transformer"]["models"]]
    print("\n================ COMPARISON ================")
    print(f"{'model':16} {'macro_f1':>9}")
    print(f"{'baseline (tfidf)':16} {'0.773':>9}  (reference)")
    for r in sorted(results, key=lambda x: -x["test_macro_f1"]):
        print(f"{r['model']:16} {r['test_macro_f1']:>9}")
    (REPORTS / "transformer_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        finetune(sys.argv[1])
    else:
        run_all()
