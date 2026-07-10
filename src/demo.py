"""
Offline demo data for the Sentiment-Forecasting pipeline.

Generates a *synthetic but realistic* market: each ticker gets a hidden
daily "news mood" that follows a slow-moving AR(1) process. That mood drives
two things at once:

1. The tone of the generated headlines for that day, and
2. A small edge in the *next day's* return.

Because the link is next-day, the pipeline's core hypothesis (today's news
sentiment predicts tomorrow's move) is actually true in the demo universe —
so every chart in the app has something interesting to show, with zero
network access, no API keys, and no model downloads. Everything is seeded,
so runs are reproducible.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

DEMO_SEED = 99

# Headline templates. The wording intentionally uses common financial
# sentiment vocabulary so both FinBERT and the lexicon fallback score them
# sensibly.
_POSITIVE_TEMPLATES = [
    "{t} beats quarterly earnings expectations as revenue surges",
    "{t} shares rally after analysts upgrade outlook",
    "{t} hits record high on strong product demand",
    "{t} announces expanded buyback and raises dividend",
    "Strong growth momentum lifts {t} to multi-month highs",
    "{t} tops profit forecasts, guidance exceeds estimates",
]
_NEGATIVE_TEMPLATES = [
    "{t} shares fall as regulators launch investigation",
    "{t} misses earnings estimates, stock drops in late trading",
    "Analysts downgrade {t} on weak demand fears",
    "{t} warns of slowdown as losses widen",
    "{t} tumbles amid broad market selloff",
    "Lawsuit risk weighs on {t} as shares decline",
]
_NEUTRAL_TEMPLATES = [
    "{t} to report quarterly results next week",
    "{t} appoints new board member",
    "What analysts are watching in {t} this quarter",
    "{t} schedules annual shareholder meeting",
    "{t} files routine quarterly statement with SEC",
]
_SOURCES = ["Market Daily", "Finance Wire", "The Street Beat", "Capital Journal"]


def _latent_sentiment(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Slow-moving AR(1) mood series clipped to [-1, 1]."""
    s = np.zeros(n_days)
    for i in range(1, n_days):
        s[i] = 0.7 * s[i - 1] + rng.normal(0, 0.35)
    return np.clip(s, -1.0, 1.0)


def generate_demo_data(
    tickers: list[str],
    news_days: int = 5,
    price_lookback_days: int = 180,
    seed: int = DEMO_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a consistent synthetic (news, close-prices) pair.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to simulate
    news_days : int
        How many recent days get headlines (mirrors the live news lookback)
    price_lookback_days : int
        Length of the simulated price history
    seed : int
        RNG seed; same seed → same demo universe

    Returns
    -------
    tuple
        (news_df, close_df) where news_df matches the live headline schema
        [ticker, ts, headline, source, link] and close_df is a business-day
        indexed frame of close prices, one column per ticker.
    """
    rng = np.random.default_rng(seed)

    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=price_lookback_days)
    n = len(dates)

    closes = {}
    news_rows = []

    for ticker in tickers:
        mood = _latent_sentiment(n, rng)

        # Next-day returns carry a small sentiment edge: r[t] responds to
        # yesterday's mood, which is exactly the relationship the pipeline
        # is built to detect.
        drift = rng.normal(0.0004, 0.0002)
        vol = rng.uniform(0.010, 0.022)
        eps = rng.normal(0, vol, n)
        alpha = 0.006
        rets = drift + eps
        rets[1:] += alpha * mood[:-1]

        px = 100.0 * np.exp(rng.normal(0, 0.5)) * np.cumprod(1.0 + rets)
        closes[ticker] = pd.Series(px, index=dates)

        # Headlines for the most recent `news_days` calendar days
        for day_idx in range(max(0, n - news_days), n):
            day = dates[day_idx]
            m = mood[day_idx]
            p_pos = float(np.clip(0.45 + 0.4 * m, 0.05, 0.9))
            p_neg = float(np.clip(0.45 - 0.4 * m, 0.05, 0.9))
            p_neu = max(0.0, 1.0 - p_pos - p_neg)
            probs = np.array([p_pos, p_neg, p_neu])
            probs = probs / probs.sum()

            for _ in range(int(rng.integers(2, 6))):
                kind = rng.choice(3, p=probs)
                pool = (_POSITIVE_TEMPLATES, _NEGATIVE_TEMPLATES, _NEUTRAL_TEMPLATES)[kind]
                template = pool[int(rng.integers(0, len(pool)))]
                ts = day + pd.Timedelta(
                    hours=int(rng.integers(11, 22)), minutes=int(rng.integers(0, 60))
                )
                news_rows.append(
                    {
                        "ticker": ticker,
                        "ts": ts.tz_localize("UTC"),
                        "headline": template.format(t=ticker),
                        "source": _SOURCES[int(rng.integers(0, len(_SOURCES)))],
                        "link": "https://example.com/demo",
                    }
                )

    close_df = pd.DataFrame(closes)
    news_df = (
        pd.DataFrame(news_rows)
        .sort_values(["ticker", "ts"])
        .reset_index(drop=True)
    )
    return news_df, close_df
