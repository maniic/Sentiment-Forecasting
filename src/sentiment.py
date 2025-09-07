from __future__ import annotations

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Pretrained finance sentiment model (FinBERT)
FINBERT_ID = "yiyanghkust/finbert-tone"

# Cache the pipeline so we load the model only once
_FINBERT_PIPE = None

def get_finbert(device: int | None = None):
    """Load a FinBERT pipeline"""
    global _FINBERT_PIPE
    
    if _FINBERT_PIPE is not None:
        return _FINBERT_PIPE
    
    # Auto-detect device if not provided
    if device is None:
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except Exception:
            device = -1 # default to CPU
    
    tok = AutoTokenizer.from_pretrained(FINBERT_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(FINBERT_ID)
    _FINBERT_PIPE = pipeline("text-classification", model=mdl, tokenizer=tok, device=device)
    
    return _FINBERT_PIPE

def score_headlines(df: pd.DataFrame, text_col: str = "headline", batch_size: int = 32) -> pd.DataFrame:
    """Add sentiment columns:
        sent_label: POSITIVE / NEGATIVE / NEUTRAL
        sent_prob: model confidence for that label (0..1)
        sent_score: signed score in [-1, 1] (POSITIVE=+prob, NEGATIVE=-prob, NEUTRAL=0)
        
    Return a copy of df with new columns. If df is empty, returns an empty-augmented df.
    """
    if df is None or df.empty:
        out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        for c in ["sent_label", "sent_prob", "sent_score"]:
            out[c] = pd.Series(dtype="float64") if c != "sent_label" else pd.Series(dtype="object")
        return out
    
    pipe = get_finbert()
    texts = df[text_col].astype(str).tolist()
    
    # Run model in batches to avoid memory spikes
    preds = pipe(texts, batch_size=batch_size)
    
    labels, probs, signed = [], [], []
    for p in preds:
        lab = p["label"].upper()
        prob = float(p["score"])
        labels.append(lab)
        probs.append(prob)
        if "POS" in lab:
            signed.append(+prob)
        elif "NEG" in lab:
            signed.append(-prob)
        else:
            signed.append(0.0)
    
    out = df.copy()
    out["sent_label"] = labels
    out["sent_prob"] = probs
    out["sent_score"] = signed
    return out

def aggregate_daily_sentiment(
    df_scored: pd.DataFrame,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Aggregate to trading-day level per ticker:
      - sent_mean: average of sent_score
      - sent_std: std dev of sent_score (0 if only one headline)
      - n: count of headlines
    """
    if df_scored is None or df_scored.empty:
        return pd.DataFrame(columns=["ticker", "date", "sent_mean", "sent_std", "n"])

    # Convert UTC timestamp → local trading date
    local_dates = df_scored["ts"].dt.tz_convert(tz).dt.date
    df_local = df_scored.assign(date=pd.to_datetime(local_dates))

    agg = (
        df_local
        .groupby(["ticker", "date"])
        .agg(
            sent_mean=("sent_score", "mean"),
            sent_std=("sent_score", lambda s: s.std(ddof=0) if len(s) > 1 else 0.0),
            n=("sent_score", "size"),
        )
        .reset_index()
    )
    return agg[["ticker", "date", "sent_mean", "sent_std", "n"]]