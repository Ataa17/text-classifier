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
| 1 | Scrape 10–20k raw comments (YouTube Data API, Reddit, news sites) | scaffolded |
| 2 | Clean: dedup, Arabic normalization, script detection (kept as a feature), keep emojis | scaffolded |
| 3 | Hand-label ~1.5–2k; 200 double-labeled for Cohen's κ; optional LLM pre-labeling | planned |
| 4 | Models: TF-IDF char n-grams + LogReg baseline → fine-tune DziriBERT / MARBERT / CAMeLBERT | planned |
| 5 | Error analysis (by phenomenon + by script), Gradio demo, write-up | planned |

Primary metric: **macro-F1** (not accuracy — classes are imbalanced by topic). Stratified split, fixed seeds.

## Data collection ethics

This project scrapes public comments for **research**. We:

- Use **official APIs** where available (YouTube Data API v3, Reddit API) — the cleanest legal route.
- Check each site's **Terms of Service and `robots.txt`**, and **rate-limit** all requests.
- Store **no usernames or personal information** — only comment text and non-identifying metadata
  (platform, language/script, timestamp bucket).
- Keep raw scraped data **out of version control** (see `.gitignore`).

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then fill in your API keys
# For JS-heavy news pages (optional):
playwright install chromium
```

Edit `config/config.yaml` to add YouTube channel IDs, subreddits, and news sites.

## Usage

```bash
# Step 1 — scrape (writes to data/raw/)
python -m src.scraping.youtube
python -m src.scraping.reddit

# Step 2 — clean (data/raw/ -> data/processed/)
python -m src.cleaning.clean
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
app/               gradio_app.py (demo)
tests/
```

## License

For research / educational use.
