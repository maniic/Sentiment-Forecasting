"""
Feature engineering for the Sentiment-Forecasting pipeline.

Computes price-based features, topology features, and joins with sentiment data.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import TopoConfig
from src.schemas import Columns, TOPO_FEATURE_COLUMNS
from src.tda import (
    compute_topo_features_for_series,
    merge_topology_features,
    topo_feature_names,
)

logger = logging.getLogger(__name__)


def _as_list(tickers: Iterable[str]) -> list[str]:
    """Convert tickers to list format."""
    if isinstance(tickers, str):
        return [tickers]
    return list(tickers)


def _download_prices(tickers: list[str], lookback_days: int) -> pd.DataFrame:
    """
    Download OHLCV data from yfinance.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols
    lookback_days : int
        Number of days of history to fetch

    Returns
    -------
    pd.DataFrame
        OHLCV data from yfinance
    """
    utc_now = pd.Timestamp.utcnow()
    now_ny = utc_now.tz_convert("America/New_York")
    start = (now_ny - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    logger.debug("Downloading prices for %s from %s", tickers, start)
    ohlcv = yf.download(tickers, start=start, auto_adjust=False, progress=False)
    return ohlcv


def _extract_close_prices(ohlcv: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Extract Adj Close prices from yfinance response.

    Handles both single-ticker and multi-ticker response formats.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Raw yfinance download response
    tickers : list[str]
        List of ticker symbols

    Returns
    -------
    pd.DataFrame
        DataFrame with tickers as columns and dates as index
    """
    if ("Adj Close" in ohlcv.columns) and isinstance(ohlcv.columns, pd.MultiIndex):
        return ohlcv["Adj Close"]
    elif "Adj Close" in ohlcv.columns:
        return ohlcv[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
    else:
        raise ValueError("Could not locate 'Adj Close' in yfinance response")


def _compute_single_ticker_features(px: pd.Series, ticker: str) -> pd.DataFrame:
    """
    Compute price features for a single ticker.

    Features computed:
    - ret_1, ret_5, ret_10, ret_20: Period returns
    - vol_20: 20-day rolling volatility
    - next_ret: Next-day return (label target)
    - label: Binary label (1 if next_ret > 0)

    Parameters
    ----------
    px : pd.Series
        Adjusted close prices
    ticker : str
        Ticker symbol

    Returns
    -------
    pd.DataFrame
        Feature DataFrame for the ticker
    """
    px = px.dropna()
    if px.empty:
        return pd.DataFrame()

    daily_ret = px.pct_change()
    feat = pd.DataFrame(
        {
            Columns.DATE: pd.to_datetime(px.index.date),
            Columns.TICKER: ticker,
            Columns.RET_1: px.pct_change().values,
            Columns.RET_5: px.pct_change(5).values,
            Columns.RET_10: px.pct_change(10).values,
            Columns.RET_20: px.pct_change(20).values,
            Columns.VOL_20: daily_ret.rolling(20).std().values,
            Columns.NEXT_RET: px.pct_change().shift(-1).values,
        }
    )

    # Drop rows with NaN in key features
    feat = feat.dropna(subset=[Columns.RET_5, Columns.RET_10, Columns.VOL_20, Columns.NEXT_RET])
    return feat


def _add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary classification labels based on next-day returns."""
    df = df.copy()
    df[Columns.LABEL] = (df[Columns.NEXT_RET] > 0).astype(int)
    return df


def _merge_topology_if_available(
    price_df: pd.DataFrame,
    topo_frames: list[pd.DataFrame],
    add_topology: bool,
) -> pd.DataFrame:
    """
    Merge topology features if computed, or add zero-filled columns.

    Parameters
    ----------
    price_df : pd.DataFrame
        Base price features DataFrame
    topo_frames : list[pd.DataFrame]
        List of topology feature DataFrames
    add_topology : bool
        Whether topology was requested

    Returns
    -------
    pd.DataFrame
        DataFrame with topology columns added
    """
    if add_topology and topo_frames:
        topo_all = pd.concat(topo_frames, ignore_index=True)
        return merge_topology_features(price_df, topo_all)
    elif add_topology:
        # Ensure topo columns exist even if computation failed
        for col in topo_feature_names():
            price_df[col] = 0.0
    return price_df


def _empty_price_features() -> pd.DataFrame:
    """Return an empty DataFrame with correct price feature schema."""
    columns = [
        Columns.TICKER,
        Columns.DATE,
        Columns.RET_1,
        Columns.RET_5,
        Columns.RET_10,
        Columns.RET_20,
        Columns.VOL_20,
        Columns.NEXT_RET,
        Columns.LABEL,
    ]
    return pd.DataFrame(columns=columns)


def compute_price_features(
    tickers: Iterable[str],
    lookback_days: int = 100,
    add_topology: bool = True,
    topo_window: int = 60,
    topo_embed_dim: int = 4,
) -> pd.DataFrame:
    """
    Download prices and compute per (ticker, date) features.

    Features include:
    - ret_1, ret_5, ret_10, ret_20: Period returns
    - vol_20: 20-day rolling volatility
    - Optional topology features (if add_topology=True)
    - next_ret: Next-day return
    - label: Binary label (1 if next_ret > 0)

    Parameters
    ----------
    tickers : Iterable[str]
        Ticker symbols to process
    lookback_days : int
        Days of price history to download
    add_topology : bool
        Whether to compute TDA features
    topo_window : int
        Window size for topology computation
    topo_embed_dim : int
        Embedding dimension for delay embedding

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with one row per (ticker, date)
    """
    tickers = _as_list(tickers)

    try:
        ohlcv = _download_prices(tickers, lookback_days)
        close = _extract_close_prices(ohlcv, tickers)
    except Exception as e:
        logger.warning("Failed to download prices: %s", e)
        return _empty_price_features()

    records = []
    topo_frames = []

    for t in close.columns:
        px = close[t]

        # Compute price features
        feat = _compute_single_ticker_features(px, t)
        if feat.empty:
            continue

        feat = _add_labels(feat)
        records.append(feat)

        # Compute topology features
        if add_topology:
            try:
                topo_cfg = TopoConfig(window=topo_window, embed_dim=topo_embed_dim)
                topo_df = compute_topo_features_for_series(px.dropna(), ticker=t, cfg=topo_cfg)
                if not topo_df.empty:
                    topo_frames.append(topo_df)
            except Exception as e:
                logger.warning("Topology computation failed for %s: %s", t, e)

    if not records:
        return _empty_price_features()

    out = pd.concat(records, ignore_index=True)
    out = out.sort_values([Columns.TICKER, Columns.DATE]).reset_index(drop=True)
    out = _merge_topology_if_available(out, topo_frames, add_topology)

    return out.reset_index(drop=True)


def join_sentiment_with_prices(
    daily_sentiment: pd.DataFrame,
    price_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join price features with daily sentiment on (ticker, date).

    Parameters
    ----------
    daily_sentiment : pd.DataFrame
        Daily aggregated sentiment with columns [ticker, date, sent_mean, sent_std, n]
    price_features : pd.DataFrame
        Price features DataFrame

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all features
    """
    if price_features is None or price_features.empty:
        return pd.DataFrame()

    sent = (
        daily_sentiment
        if daily_sentiment is not None
        else pd.DataFrame(
            columns=[Columns.TICKER, Columns.DATE, Columns.SENT_MEAN, Columns.SENT_STD, Columns.N_HEADLINES]
        )
    )

    merged = price_features.merge(
        sent,
        on=[Columns.TICKER, Columns.DATE],
        how="left",
    ).fillna({Columns.SENT_MEAN: 0.0, Columns.SENT_STD: 0.0, Columns.N_HEADLINES: 0})

    # Build column list preserving order
    base_cols = [
        Columns.TICKER,
        Columns.DATE,
        Columns.RET_1,
        Columns.RET_5,
        Columns.RET_10,
        Columns.RET_20,
        Columns.VOL_20,
        Columns.SENT_MEAN,
        Columns.SENT_STD,
        Columns.N_HEADLINES,
        Columns.NEXT_RET,
        Columns.LABEL,
    ]
    topo_cols = [c for c in merged.columns if c.startswith("topo_")]
    keep = [c for c in base_cols if c in merged.columns] + topo_cols

    merged[topo_cols] = merged[topo_cols].fillna(0.0)

    return merged[keep].sort_values([Columns.TICKER, Columns.DATE]).reset_index(drop=True)
