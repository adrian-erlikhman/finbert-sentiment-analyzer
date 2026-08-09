# 📰 FinBERT Market-Sentiment Analyzer

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white) ![Transformers](https://img.shields.io/badge/🤗-Transformers-FFD21E) ![License](https://img.shields.io/badge/license-MIT-black)

Turns financial headlines into a daily sentiment signal with **FinBERT** — plus a lexicon fallback so the pipeline runs even offline.

```mermaid
flowchart LR
    A[Headlines CSV] --> B{FinBERT<br/>available?}
    B -->|yes| C[ProsusAI/finbert]
    B -->|no| D[Lexicon fallback]
    C --> E[pos / neg / neutral]
    D --> E
    E --> F[Daily aggregate signal]
```

**How** — FinBERT tags each headline, then aggregates a per-day score you can line up against price. No `transformers`/`torch`? It falls back to a finance lexicon so it always runs end to end.

## Run
```bash
pip install -r requirements.txt
python sentiment.py --csv sample_headlines.csv   # date, headline columns
```

<sub>MIT · Adrian Erlikhman</sub>
