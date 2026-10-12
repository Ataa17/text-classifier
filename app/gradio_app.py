"""Step 5 — Gradio demo.

STUB. Loads the best trained model and classifies a typed Derja comment as pos/neg,
showing the detected script alongside. Wire up in step 5.

Run:  python -m app.gradio_app
"""
from __future__ import annotations

import gradio as gr

from src.cleaning.normalize import normalize
from src.cleaning.script_detect import detect_script


def predict(text: str):
    clean = normalize(text)
    script = detect_script(clean)
    # TODO (step 5): load trained model and return a real prediction.
    return {"(model not trained yet)": 1.0}, f"script: {script}"


def build():
    return gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=3, label="Tunisian Derja comment"),
        outputs=[gr.Label(label="Sentiment"), gr.Textbox(label="Detected script")],
        title="Tunisian Derja Sentiment (pos/neg)",
        description="Demo — model training lands in step 4.",
    )


if __name__ == "__main__":
    build().launch()
