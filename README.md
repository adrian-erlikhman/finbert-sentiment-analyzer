# FinBERT Market-Sentiment Analyzer

An NLP pipeline that classifies **financial-headline sentiment** with
**FinBERT** (`ProsusAI/finbert`) and aggregates a **daily sentiment signal** you
can align against market moves.

## What it does
- Runs headlines through FinBERT for positive / negative / neutral labels
- Maps labels to a numeric score and aggregates a per-day signal
- **Graceful fallback:** if `transformers`/`torch` aren't installed (or you're
  offline), it uses a compact finance lexicon so the pipeline still runs

## Run it
```bash
pip install -r requirements.txt
python sentiment.py --csv sample_headlines.csv
```
Bring your own headlines with any CSV that has `date` and `headline` columns.

## Notes
Ships with a small sample headline set. The lexicon fallback exists so the
project is always runnable in one command; install `transformers` + `torch` to
switch on the real FinBERT model.

_Author: Adrian Erlikhman_
