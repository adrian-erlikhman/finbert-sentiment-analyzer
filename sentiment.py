"""
FinBERT Market-Sentiment Analyzer
=================================
Classifies the sentiment (positive / negative / neutral) of financial headlines
using FinBERT (ProsusAI/finbert) and aggregates a daily sentiment signal that
can be lined up against price action.

If `transformers`/`torch` or the model download aren't available, it falls back
to a small finance lexicon so the full pipeline still runs end-to-end.
    python sentiment.py --csv sample_headlines.csv
"""
import argparse
import pandas as pd

POS = {"beat", "beats", "surge", "surges", "soars", "gains", "record", "upgrade",
       "growth", "profit", "rally", "strong", "tops", "jumps", "rebound"}
NEG = {"miss", "misses", "plunge", "plunges", "falls", "drop", "cut", "cuts",
       "downgrade", "loss", "weak", "lawsuit", "probe", "slump", "warns", "fears"}


def lexicon_score(text):
    words = text.lower().replace(",", " ").split()
    net = sum(w in POS for w in words) - sum(w in NEG for w in words)
    return "positive" if net > 0 else "negative" if net < 0 else "neutral"


def load_finbert():
    try:
        from transformers import pipeline
        return pipeline("text-classification", model="ProsusAI/finbert")
    except Exception as exc:  # no transformers/torch, or offline
        print(f"[fallback] FinBERT unavailable ({exc.__class__.__name__}); "
              f"using lexicon scorer instead.")
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="sample_headlines.csv")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    pipe = load_finbert()

    if pipe is not None:
        df["sentiment"] = [r["label"].lower() for r in pipe(df["headline"].tolist())]
    else:
        df["sentiment"] = [lexicon_score(h) for h in df["headline"]]

    df["score"] = df["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1})

    print(df[["date", "headline", "sentiment"]].to_string(index=False))
    print("\n--- Daily aggregate sentiment signal ---")
    print(df.groupby("date")["score"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
