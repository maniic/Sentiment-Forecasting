"""
Sentiment analysis using FinBERT for financial headlines.

Provides thread-safe access to the FinBERT model and functions for
scoring headlines and aggregating daily sentiment.
"""
from __future__ import annotations

import logging
import threading

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from src.config import MARKET_TZ
from src.schemas import Columns

logger = logging.getLogger(__name__)

# Pretrained finance sentiment model (FinBERT)
FINBERT_ID = "yiyanghkust/finbert-tone"

# Thread-safe singleton pattern for FinBERT pipeline
_FINBERT_PIPE = None
_FINBERT_LOCK = threading.Lock()


def get_finbert(device: int | None = None):
    """
    Load a FinBERT pipeline with thread-safe singleton pattern.

    Uses double-checked locking to ensure the model is only loaded once
    even when called from multiple threads.

    Parameters
    ----------
    device : int | None
        Device index (0 for CUDA, -1 for CPU, None for auto-detect)

    Returns
    -------
    Pipeline
        Hugging Face text-classification pipeline
    """
    global _FINBERT_PIPE

    # Fast path: return if already initialized
    if _FINBERT_PIPE is not None:
        return _FINBERT_PIPE

    # Thread-safe initialization with double-checked locking
    with _FINBERT_LOCK:
        # Check again inside lock
        if _FINBERT_PIPE is not None:
            return _FINBERT_PIPE

        # Auto-detect device if not provided
        if device is None:
            try:
                import torch

                device = 0 if torch.cuda.is_available() else -1
            except Exception:
                device = -1  # default to CPU

        logger.info("Loading FinBERT model (device=%s)...", device)
        tok = AutoTokenizer.from_pretrained(FINBERT_ID)
        mdl = AutoModelForSequenceClassification.from_pretrained(FINBERT_ID)
        _FINBERT_PIPE = pipeline("text-classification", model=mdl, tokenizer=tok, device=device)
        logger.info("FinBERT model loaded successfully")

        return _FINBERT_PIPE


def score_headlines(
    df: pd.DataFrame,
    text_col: str = "headline",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Add sentiment columns to headlines DataFrame.

    Columns added:
    - sent_label: POSITIVE / NEGATIVE / NEUTRAL
    - sent_prob: Model confidence for that label (0..1)
    - sent_score: Signed score in [-1, 1]
      (POSITIVE=+prob, NEGATIVE=-prob, NEUTRAL=0)

    Parameters
    ----------
    df : pd.DataFrame
        Headlines DataFrame with text column
    text_col : str
        Name of column containing text to score
    batch_size : int
        Batch size for model inference

    Returns
    -------
    pd.DataFrame
        Copy of input with sentiment columns added
    """
    if df is None or df.empty:
        out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        for c in [Columns.SENT_LABEL, Columns.SENT_PROB, Columns.SENT_SCORE]:
            out[c] = pd.Series(dtype="float64") if c != Columns.SENT_LABEL else pd.Series(dtype="object")
        return out

    pipe = get_finbert()
    texts = df[text_col].astype(str).tolist()

    logger.debug("Scoring %d headlines", len(texts))

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
    out[Columns.SENT_LABEL] = labels
    out[Columns.SENT_PROB] = probs
    out[Columns.SENT_SCORE] = signed
    return out


def aggregate_daily_sentiment(
    df_scored: pd.DataFrame,
    tz: str = MARKET_TZ,
) -> pd.DataFrame:
    """
    Aggregate scored headlines to trading-day level per ticker.

    Computes:
    - sent_mean: Average of sent_score
    - sent_std: Standard deviation of sent_score (0 if only one headline)
    - n: Count of headlines

    Parameters
    ----------
    df_scored : pd.DataFrame
        Scored headlines with sent_score column
    tz : str
        Timezone for converting UTC timestamps to trading dates

    Returns
    -------
    pd.DataFrame
        Daily aggregated sentiment with columns [ticker, date, sent_mean, sent_std, n]
    """
    if df_scored is None or df_scored.empty:
        return pd.DataFrame(
            columns=[Columns.TICKER, Columns.DATE, Columns.SENT_MEAN, Columns.SENT_STD, Columns.N_HEADLINES]
        )

    # Convert UTC timestamp → local trading date
    local_dates = df_scored[Columns.TIMESTAMP].dt.tz_convert(tz).dt.date
    df_local = df_scored.assign(**{Columns.DATE: pd.to_datetime(local_dates)})

    agg = (
        df_local.groupby([Columns.TICKER, Columns.DATE])
        .agg(
            **{
                Columns.SENT_MEAN: (Columns.SENT_SCORE, "mean"),
                Columns.SENT_STD: (Columns.SENT_SCORE, lambda s: s.std(ddof=0) if len(s) > 1 else 0.0),
                Columns.N_HEADLINES: (Columns.SENT_SCORE, "size"),
            }
        )
        .reset_index()
    )

    return agg[[Columns.TICKER, Columns.DATE, Columns.SENT_MEAN, Columns.SENT_STD, Columns.N_HEADLINES]]
