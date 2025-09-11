from __future__ import annotations

import math
import numpy as np
import pandas as pd

def apply_signals(df: pd.DataFrame, p_col: str = "p_up", threshold: float = 0.60, require_mom_agree: bool = True, mom_col: str = "ret_5") -> pd.DataFrame:
    """Add a boolean 'long' column:
        long = (p_up > threshold) [& (ret_5 > 0 if require_mom_agree)]
    """
    out = df.copy()
    long = out[p_col] > float(threshold)
    if require_mom_agree:
        long &= out[mom_col].astype(float) > 0
    out["long"] = long.astype(bool)
    return out


def backtest_equal_weight(df_signals: pd.DataFrame, cost_bps: float = 3.0) -> tuple[pd.DataFrame, dict]:
    """Daily rebalance: equal-weight among 'long' tickers
    Uses next_ret as the realized next -day return for signals formed at 'date'.
    
    Cost model (MVP): subtract one round-trip cost per trading day with any positions.
    """
    if df_signals.empty:
        return pd.DataFrame(), {"sharpe": np.nan, "max_dd": np.nan, "trades": 0}
    
    df = df_signals.sort_values(["date", "ticker"]).copy()
    
    daily = []
    for d, g in df.groupby("date", sort=True):
        longs = g[g["long"]]
        if longs.empty:
            daily.append({"date": d, "n_positions": 0, "gross": 0.0, "net": 0.0})
            continue
        
        gross = float(longs["next_ret"].mean())  # equal weight
        cost = (cost_bps / 10000.0)   # one round-trip per day we trade
        net = gross - cost
        daily.append({"date": d, "n_positions": int(len(longs)), "gross": gross, "net": net})
        
    perf = pd.DataFrame(daily).sort_values("date").reset_index(drop=True)
    perf["equity"]= (1.0 + perf["net"]).cumprod()
    
    # Metrics
    ret = perf["net"].values
    trades = int((perf["n_positions"] > 0).sum())
    sharpe = np.nan
    if np.nanstd(ret) > 0:
        sharpe = np.sqrt(252.0) * np.nanmean(ret) / np.nanstd(ret)
        
    # Max drawdown
    eq = perf["equity"].values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min()) if len(dd) else np.nan

    metrics = {"sharpe": float(sharpe), "max_dd": max_dd, "trades": trades}
    return perf, metrics

def todays_signals(df_with_p: pd.DataFrame, threshold: float = 0.60, require_mom_agree: bool = True) -> pd.DataFrame:
    """"Return long candidates on the most recent date in df_with_p
    Columns expected: date, p_up, ret_5, sent_mean, n, ticker"""
    if df_with_p is None or df_with_p.empty:
        return pd.DataFrame(columns=["ticker", "p_up", "ret_5", "sent_mean", "n", "date"])
    
    last_date = df_with_p["date"].max()
    day = df_with_p[df_with_p["date"] == last_date].copy()
    
    mask = day["p_up"] > float(threshold)
    if require_mom_agree:
        mask &= day["ret_5"] > 0.0
        
    picks = day[mask].sort_values("p_up", ascending=False)
    return picks[["ticker", "p_up", "ret_5", "sent_mean", "n", "date"]].reset_index(drop=True)