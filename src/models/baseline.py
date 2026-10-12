"""Step 4 baseline — TF-IDF char n-grams + Logistic Regression.

Char n-grams are robust to the heavy misspelling / spelling variation in Arabizi.
STUB: the pipeline is sketched; wire it to labeled data in step 4.

Run (once labeled data exists):  python -m src.models.baseline
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5),
            min_df=2, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed,
        )),
    ])


def main():
    raise NotImplementedError("Load data/labeled/, split, fit build_pipeline(), report macro-F1.")


if __name__ == "__main__":
    main()
