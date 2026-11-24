"""Small Streamlit interface for the three trained sentiment models."""
from __future__ import annotations

import joblib
import streamlit as st
import torch

from src.cleaning.normalize import normalize
from src.cleaning.script_detect import detect_script


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

MODEL_OPTIONS = {
    "TF-IDF": "tfidf",
    "TunBERT": "tunbert",
    "DziriBERT": "dziribert",
}


@st.cache_resource
def load_model(model_name: str):
    if model_name == "TF-IDF":
        return joblib.load(ROOT / "models" / "baseline.joblib"), "tfidf"

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = ROOT / "models" / model_name.lower()
    trust_remote_code = model_name == "TunBERT"
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return (tokenizer, model, device), "transformer"


def predict_tfidf(model, text: str) -> tuple[str, dict[str, float]]:
    probabilities = model.predict_proba([text])[0]
    labels = model.classes_
    scores = {str(label): float(score) for label, score in zip(labels, probabilities)}
    predicted = max(scores, key=scores.get)
    return predicted, scores


def predict_transformer(model_bundle, text: str) -> tuple[str, dict[str, float]]:
    tokenizer, model, device = model_bundle
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

    id2label = getattr(model.config, "id2label", None) or {
        0: "negative",
        1: "positive",
    }
    scores = {
        str(id2label.get(index, id2label.get(str(index), str(index)))): float(score)
        for index, score in enumerate(probabilities)
    }
    predicted = max(scores, key=scores.get)
    return predicted, scores


st.set_page_config(page_title="Derja Sentiment", page_icon="TN")
st.title("Tunisian Derja Sentiment")
st.caption("Positive / negative sentiment classifier")

model_name = st.selectbox("Model", list(MODEL_OPTIONS))
text = st.text_area(
    "Tunisian Derja text",
    placeholder="Example: behi barcha / تحب تونس",
    height=140,
)

if st.button("Predict", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Enter some text to classify.")
    else:
        clean = normalize(text)
        script = detect_script(clean)
        model, model_type = load_model(model_name)

        if model_type == "tfidf":
            predicted, scores = predict_tfidf(model, clean)
        else:
            predicted, scores = predict_transformer(model, clean)

        st.subheader("Prediction")
        st.metric("Label", predicted)
        st.progress(max(scores.values()), text="Highest confidence")
        st.write("**Detected script:** " + script)
        st.write("**Normalized text:** " + (clean or "(empty)"))
        st.json(scores)
