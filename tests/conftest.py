"""
Pytest fixtures for Sentiment-Forecasting tests.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_news_df():
    """Sample news DataFrame for testing."""
    return pd.DataFrame({
        "ticker": ["AAPL", "AAPL", "MSFT"],
        "ts": pd.to_datetime([
            "2024-01-15 10:00:00",
            "2024-01-15 14:00:00",
            "2024-01-15 11:00:00",
        ], utc=True),
        "headline": [
            "Apple announces new product line",
            "AAPL stock rises on strong earnings",
            "Microsoft cloud revenue beats expectations",
        ],
        "source": ["Reuters", "Bloomberg", "CNBC"],
        "link": ["http://example.com/1", "http://example.com/2", "http://example.com/3"],
    })


@pytest.fixture
def sample_scored_df(sample_news_df):
    """Sample scored news DataFrame."""
    df = sample_news_df.copy()
    df["sent_label"] = ["POSITIVE", "POSITIVE", "POSITIVE"]
    df["sent_prob"] = [0.85, 0.90, 0.75]
    df["sent_score"] = [0.85, 0.90, 0.75]
    return df


@pytest.fixture
def sample_daily_sentiment():
    """Sample daily sentiment DataFrame."""
    return pd.DataFrame({
        "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "date": pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-15", "2024-01-16"]),
        "sent_mean": [0.5, 0.3, -0.2, 0.1],
        "sent_std": [0.1, 0.2, 0.15, 0.05],
        "n": [5, 3, 4, 2],
    })


@pytest.fixture
def sample_price_features():
    """Sample price features DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    tickers = ["AAPL", "MSFT"]

    records = []
    for ticker in tickers:
        for date in dates:
            records.append({
                "ticker": ticker,
                "date": date,
                "ret_1": np.random.normal(0, 0.02),
                "ret_5": np.random.normal(0, 0.04),
                "ret_10": np.random.normal(0, 0.06),
                "ret_20": np.random.normal(0, 0.08),
                "vol_20": abs(np.random.normal(0.02, 0.005)),
                "next_ret": np.random.normal(0, 0.02),
                "label": np.random.choice([0, 1]),
                "topo_d0_total": np.random.uniform(0, 1),
                "topo_d1_total": np.random.uniform(0, 0.5),
                "topo_entropy": np.random.uniform(0, 0.1),
            })

    return pd.DataFrame(records)


@pytest.fixture
def sample_joined_features(sample_price_features, sample_daily_sentiment):
    """Sample joined features DataFrame."""
    df = sample_price_features.merge(
        sample_daily_sentiment,
        on=["ticker", "date"],
        how="left"
    ).fillna({"sent_mean": 0.0, "sent_std": 0.0, "n": 0})
    return df
