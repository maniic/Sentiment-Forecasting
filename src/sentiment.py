"""
Sentiment analysis for financial headlines.

Two engines are supported:

1. **FinBERT** (``yiyanghkust/finbert-tone``) — a transformer fine-tuned on
   financial text. Used automatically when ``transformers``/``torch`` are
   installed. Loaded lazily and cached with a thread-safe singleton so the
   ~400MB model is only pulled once, and only when actually needed.

2. **Lexicon fallback** — a lightweight financial word-list scorer used when
   the deep-learning stack is unavailable (e.g. a laptop without torch, or a
   free hosting tier). It keeps the whole pipeline functional everywhere.

``score_headlines`` picks the best available engine and records which one it
used in ``active_engine()`` so the UI can report it honestly.
"""
from __future__ import annotations

import logging
import re
import threading

import pandas as pd

from src.config import MARKET_TZ
from src.schemas import Columns

logger = logging.getLogger(__name__)

# Pretrained finance sentiment model (FinBERT)
FINBERT_ID = "yiyanghkust/finbert-tone"

# Thread-safe singleton pattern for FinBERT pipeline
_FINBERT_PIPE = None
_FINBERT_LOCK = threading.Lock()

# Which engine scored the most recent batch ("finbert" | "lexicon" | None)
_ACTIVE_ENGINE: str | None = None


def active_engine() -> str | None:
    """Name of the engine used by the most recent ``score_headlines`` call."""
    return _ACTIVE_ENGINE


def finbert_available() -> bool:
    """True if the transformers/torch stack can be imported."""
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def get_finbert(device: int | None = None):
    """
    Load a FinBERT pipeline with thread-safe singleton pattern.

    Uses double-checked locking to ensure the model is only loaded once
    even when called from multiple threads. Imports transformers lazily so
    that merely importing this module never requires torch.

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

        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

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


# ---------------------------------------------------------------------------
# Lexicon fallback engine
# ---------------------------------------------------------------------------
# Compact financial sentiment word lists (inspired by the Loughran-McDonald
# finance lexicon). Deliberately small: the goal is a transparent, dependency-
# free fallback, not to beat FinBERT.
_POSITIVE_WORDS = {
    "beat", "beats", "strong", "surge", "surges", "soar", "soars", "rally",
    "rallies", "record", "growth", "profit", "profits", "gain", "gains",
    "upgrade", "upgraded", "bullish", "outperform", "exceeds", "exceeded",
    "jump", "jumps", "rise", "rises", "rose", "boom", "breakthrough", "wins",
    "win", "success", "successful", "expands", "expansion", "optimistic",
    "buyback", "dividend", "tops", "topped", "positive", "momentum",
}
_NEGATIVE_WORDS = {
    "miss", "misses", "missed", "weak", "plunge", "plunges", "crash",
    "crashes", "fall", "falls", "fell", "drop", "drops", "dropped", "loss",
    "losses", "downgrade", "downgraded", "bearish", "underperform", "cuts",
    "cut", "layoff", "layoffs", "lawsuit", "probe", "investigation", "fraud",
    "recall", "warning", "warns", "slump", "slumps", "tumble", "tumbles",
    "decline", "declines", "fears", "fear", "risk", "selloff", "negative",
    "bankruptcy", "default", "slowdown",
}

_WORD_RE = re.compile(r"[a-z']+")


def _lexicon_score(text: str) -> tuple[str, float, float]:
    """
    Score one headline with the word-list engine.

    Returns
    -------
    tuple
        (label, prob, signed_score) mirroring FinBERT's output format.
    """
    words = _WORD_RE.findall(text.lower())
    pos = sum(w in _POSITIVE_WORDS for w in words)
    neg = sum(w in _NEGATIVE_WORDS for w in words)
    hits = pos - neg
    if hits == 0:
        return "NEUTRAL", 0.5, 0.0
    # Confidence grows with the number of net sentiment words, capped at 0.95
    prob = min(0.55 + 0.15 * abs(hits), 0.95)
    if hits > 0:
        return "POSITIVE", prob, +prob
    return "NEGATIVE", prob, -prob


def _score_with_lexicon(texts: list[str]) -> tuple[list[str], list[float], list[float]]:
    labels, probs, signed = [], [], []
    for t in texts:
        lab, prob, sgn = _lexicon_score(t)
        labels.append(lab)
        probs.append(prob)
        signed.append(sgn)
    return labels, probs, signed


def _score_with_finbert(
    texts: list[str], batch_size: int
) -> tuple[list[str], list[float], list[float]]:
    pipe = get_finbert()
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
    return labels, probs, signed


def score_headlines(
    df: pd.DataFrame,
    text_col: str = "headline",
    batch_size: int = 32,
    engine: str = "auto",
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
    engine : str
        "auto" (FinBERT if installed, else lexicon), "finbert", or "lexicon"

    Returns
    -------
    pd.DataFrame
        Copy of input with sentiment columns added
    """
    global _ACTIVE_ENGINE

    if df is None or df.empty:
        out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        for c in [Columns.SENT_LABEL, Columns.SENT_PROB, Columns.SENT_SCORE]:
            out[c] = pd.Series(dtype="float64") if c != Columns.SENT_LABEL else pd.Series(dtype="object")
        return out

    texts = df[text_col].astype(str).tolist()
    logger.debug("Scoring %d headlines (engine=%s)", len(texts), engine)

    use_finbert = engine == "finbert" or (engine == "auto" and finbert_available())
    if use_finbert:
        try:
            labels, probs, signed = _score_with_finbert(texts, batch_size)
            _ACTIVE_ENGINE = "finbert"
        except Exception as e:
            if engine == "finbert":
                raise
            logger.warning("FinBERT failed (%s); falling back to lexicon engine", e)
            labels, probs, signed = _score_with_lexicon(texts)
            _ACTIVE_ENGINE = "lexicon"
    else:
        labels, probs, signed = _score_with_lexicon(texts)
        _ACTIVE_ENGINE = "lexicon"

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
