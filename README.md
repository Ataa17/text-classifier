# Tunisian Derja Sentiment Classifier

Binary **positive / negative** sentiment classification for Tunisian Arabic (Derja) — a
code-switched mix of Arabic script, Latin-script "Arabizi" (e.g. *"behi barcha"*), and French.

> **Why binary?** The public Tunisian benchmarks (TSAC, TUNIZI) are both pos/neg, so a binary
> task lets us compare directly against published results without remapping classes, and it
> removes the hardest annotation calls (the neutral boundary), raising inter-annotator agreement.
> Genuinely neutral / factual comments are **discarded**, not forced into a bucket.

## Pipeline

| Step | What | Status |
|------|------|--------|
| 1 | Scrape comments through the YouTube Data API; optional news and Reddit scrapers | YouTube working |
| 2 | Deduplication, normalization, script detection, and emoji-preserving cleaning | working |
| 3 | Hand-label positive/negative/neutral samples and LLM pre-labeling support | data labeled; kappa validation pending |
| 4 | TF-IDF baseline and CUDA transformer fine-tuning | working |
| 5 | Streamlit model demo | working |
| 6 | Error analysis and transformer comparison | partial |

Primary metric: **macro-F1** (not accuracy — classes are imbalanced by topic). Stratified split, fixed seeds.

## Verified results

The existing labeled YouTube data contains 1,948 confirmed positive/negative rows.
The held-out test split contains 390 rows.

| Model | Held-out macro-F1 |
|------|------------------:|
| TF-IDF character n-grams + Logistic Regression | 0.7727 |
| TunBERT (`tunis-ai/TunBERT`) | 0.4353 |
| DziriBERT (`alger-ia/dziribert`) | 0.8353 |

DziriBERT has the strongest measured macro-F1 so far. TF-IDF remains a strong baseline and
TunBERT did not generalize well to this held-out dataset.

Per-script DziriBERT macro-F1:

| Script | Samples | Macro-F1 |
|--------|--------:|---------:|
| Arabic | 138 | 0.8547 |
| Latin | 244 | 0.8097 |
| Mixed | 8 | 1.0000 |

## Data collection ethics

This project scrapes public comments for **research**. We:

- Use **official APIs** where available (YouTube Data API v3) — the cleanest legal route.
- Check each site's **Terms of Service and `robots.txt`**, and **rate-limit** all requests.
- Store **no usernames or personal information** — only comment text and non-identifying metadata
  (platform, language/script, timestamp bucket).
- Keep raw scraped data **out of version control** (see `.gitignore`).

> **Reddit not pursued.** Reddit disabled self-serve API app creation in its 2025 Responsible
> Builder Policy rollout, and r/Tunisia skews English/French. `src/scraping/reddit.py` is kept
> for the future but is not part of the data collection. YouTube is the primary source, with
> news-site comments secondary.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then fill in your API keys
# For JS-heavy news pages (optional):
playwright install chromium
```

For CUDA-enabled transformer training on Windows, Python 3.12 is also supported by a
separate virtual environment:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -r requirements.txt

# Example CUDA 12.8 build
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
pip install "transformers>=4.40,<5" "datasets>=2.18,<4"
```

The transformer runner verifies CUDA availability and fails clearly if a GPU is unavailable.
It can be used with an NVIDIA GPU; the TF-IDF baseline itself runs on CPU.

Edit `config/config.yaml` to set YouTube channels (by ID, `@handle`, or URL) and news sites.
A starter set of Tunisian channels is pre-filled — verify and extend it.

## Usage

```bash
# Step 1 — scrape (writes to data/raw/)
python -m src.scraping.youtube
# python -m src.scraping.news     # once news sites are configured

# Step 2 — clean (data/raw/ -> data/processed/)
python -m src.cleaning.clean

# Step 3 — labeling (binary pos/neg; neutral is discarded)
python -m src.labeling.sample          # draw kappa (200) + main (2600) sets
python -m src.labeling.llm_prelabel    # optional: LLM first-pass on the main set
python -m src.labeling.label_cli kappa # hand-label the blind kappa set first
python -m src.labeling.label_cli       # then the main set (Enter accepts a suggestion)
python -m src.labeling.kappa a.csv b.csv   # Cohen's kappa from two annotators

# Step 4 — baseline model (TF-IDF char n-grams + LogReg)
python -m src.models.baseline          # trains, reports macro-F1 + by-script, saves model

# Fine-tune transformer models with CUDA
python -m src.models.transformer tunbert
python -m src.models.transformer dziribert
# Or train every configured transformer:
python -m src.models.transformer

# Step 5 — Streamlit prediction interface
.venv312\Scripts\python.exe -m streamlit run app/streamlit_app.py

# Tests
python -m pytest -q
```

## Layout

```
config/            config.yaml (non-secret settings)
data/              raw / interim / processed / labeled  (gitignored)
src/
  config.py        loads .env + config.yaml
  scraping/        youtube.py, reddit.py, news.py
  cleaning/        normalize.py (Arabic), script_detect.py, clean.py
  labeling/        llm_prelabel.py
  models/          baseline.py, transformer.py
  evaluation/      metrics.py
app/
  gradio_app.py     legacy Gradio demo
  streamlit_app.py  model-selection prediction interface
models/             baseline and fine-tuned model artifacts (gitignored)
reports/            metrics and evaluation reports (gitignored)
tests/
```

## License

For research / educational use.
