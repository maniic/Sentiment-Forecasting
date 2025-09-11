from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd
import yfinance as yf

def _as_list(tickers: Iterable[str]) -> list[str]:
    if isinstance(tickers, str):
        return [tickers]
    return list(tickers)

def compute_price_features(tickers: Iterable[str], lookback_days: int = 100) -> pd.DataFrame:
    """Download Adj Close and compute per (ticker, date):
        - ret_5, ret_10 (pct change)
        - vol_20 (rolling std of daily returns)
        - next_ret (t -> t+1 return)
        - label (1 if next_ret > 0 else 0)
    """
    tickers = _as_list(tickers)
    utc_now = pd.Timestamp.utcnow()
    now_ny = utc_now.tz_convert("America/New_York")
    start = (now_ny - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    ohlcv = yf.download(tickers, start=start, auto_adjust=False, progress=False)
    
    # Handle single vs multi-ticker shapes
    if ("Adj Close" in ohlcv.columns) and isinstance(ohlcv.columns, pd.MultiIndex):
        close = ohlcv["Adj Close"] # columns = tickers
    elif "Adj Close" in ohlcv.columns:
        close = ohlcv[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
    else:
        raise ValueError("Could not locate 'Adj Close' in yfinance response")
    
    records = []
    for t in close.columns:
        px = close[t].dropna()
        if px.empty:
            continue
        
        daily_ret = px.pct_change()
        feat = pd.DataFrame({
            "date": pd.to_datetime(px.index.date),
            "ticker": t,
            "ret_5": px.pct_change(5).values,
            "ret_10": px.pct_change(10).values,
            "vol_20": daily_ret.rolling(20).std().values,
            "next_ret": px.pct_change().shift(-1).values,
        }).dropna(subset=["ret_5", "ret_10", "vol_20", "next_ret"])
        
        feat["label"] = (feat["next_ret"] > 0).astype(int)
        records.append(feat)
        
    if not records:
        return pd.DataFrame(columns=["ticker", "date", "ret_5", "ret_10", "vol_20", "next_ret", "label"])
    
    out = pd.concat(records, ignore_index=True)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)

def join_sentiment_with_prices(daily_sentiment: pd.DataFrame, price_features: pd.DataFrame) -> pd.DataFrame:
    """Left-join price features with daily sentiment on (ticker, date)."""
    if price_features is None or price_features.empty:
        return pd.DataFrame()
    
    sent = daily_sentiment if daily_sentiment is not None else pd.DataFrame(
        columns=["ticker", "date", "sent_mean", "sent_std", "n"]
    )
    
    merged = price_features.merge(sent, on=["ticker", "date"], how="left").fillna({"sent_mean": 0.0, "sent_std": 0.0, "n": 0})
    
    cols = ["ticker", "date", "ret_5", "ret_10", "vol_20", "sent_mean", "sent_std", "n","next_ret","label"]
    return merged[cols].sort_values(["ticker", "date"]).reset_index(drop=True)