"""
FastAPI backend for Sentiment Forecasting.

Serves the hand-built frontend from ./frontend and exposes the analysis
pipeline as a JSON API. Run with:

    python server.py                # http://localhost:8000
    uvicorn server:app --reload     # dev mode
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_LOOKBACK_DAYS,
    PREDICTION_THRESHOLD,
    ROUND_TRIP_COST_BPS,
    TICKERS,
)
from src.backtest import apply_signals, backtest_equal_weight, todays_signals
from src.demo import generate_demo_data
from src.features import (
    compute_price_features_from_closes,
    join_sentiment_with_prices,
    load_close_prices,
)
from src.ml import add_quick_prob
from src.news import fetch_all_headlines
from src.sentiment import (
    active_engine,
    aggregate_daily_sentiment,
    finbert_available,
    score_headlines,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Forecasting API", version="2.0")


class RunRequest(BaseModel):
    """Parameters for one analysis run."""

    mode: Literal["demo", "live"] = "demo"
    tickers: list[str] = Field(default=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"], min_length=1)
    news_days: int = Field(default=DEFAULT_LOOKBACK_DAYS, ge=1, le=14)
    price_lookback: int = Field(default=180, ge=60, le=365)
    threshold: float = Field(default=PREDICTION_THRESHOLD, ge=0.45, le=0.75)
    require_momentum: bool = True
    cost_bps: float = Field(default=float(ROUND_TRIP_COST_BPS), ge=0, le=100)
    engine: Literal["auto", "finbert", "lexicon"] = "lexicon"
    news_source: Literal["auto", "google", "yfinance"] = "auto"


def _clean(value):
    """Make a scalar JSON-safe (NaN/inf → None, numpy → python)."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else f
    return value


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """DataFrame → JSON-safe list of records, restricted to cols."""
    out = []
    for _, row in df[cols].iterrows():
        rec = {}
        for c in cols:
            v = row[c]
            if isinstance(v, pd.Timestamp):
                rec[c] = v.isoformat()
            else:
                rec[c] = _clean(v)
        out.append(rec)
    return out


@app.get("/api/config")
def get_config():
    """Frontend bootstrap: ticker universe, defaults, capabilities."""
    return {
        "tickers": TICKERS,
        "presets": {
            "Mega Tech": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
            "Top ETFs": ["SPY", "QQQ"],
        },
        "defaults": {
            "threshold": float(PREDICTION_THRESHOLD),
            "news_days": int(DEFAULT_LOOKBACK_DAYS),
            "price_lookback": 180,
            "cost_bps": float(ROUND_TRIP_COST_BPS),
        },
        "finbert_available": finbert_available(),
    }


@app.post("/api/run")
def run_analysis(req: RunRequest):
    """Execute the full pipeline and return everything the dashboard needs."""
    tickers = [t.upper().strip() for t in req.tickers if t.strip()]

    # 1) Headlines + prices
    if req.mode == "demo":
        news, close = generate_demo_data(
            tickers, news_days=req.news_days, price_lookback_days=req.price_lookback
        )
    else:
        news = fetch_all_headlines(tickers, req.news_days, source=req.news_source)
        try:
            close = load_close_prices(tickers, lookback_days=req.price_lookback)
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Could not download prices ({e}). Try demo mode.",
            )

    if not news.empty:
        news["ts"] = pd.to_datetime(news["ts"], utc=True, errors="coerce")

    # 2) Sentiment → features → probabilities → signals → backtest
    scored = score_headlines(news, engine=req.engine)
    daily = aggregate_daily_sentiment(scored)
    price = compute_price_features_from_closes(close)
    X = join_sentiment_with_prices(daily, price)

    if X.empty:
        raise HTTPException(
            status_code=502,
            detail="No data came back — live news/price sources may be unreachable. "
            "Try demo mode.",
        )

    Xp = add_quick_prob(X)
    Xs = apply_signals(Xp, threshold=req.threshold, require_mom_agree=req.require_momentum)
    perf, metrics = backtest_equal_weight(Xs, cost_bps=req.cost_bps)
    picks = todays_signals(Xp, threshold=req.threshold, require_mom_agree=req.require_momentum)

    # 3) Shape the response
    perf = perf.copy()
    perf["date"] = pd.to_datetime(perf["date"])

    sent_by_ticker = (
        daily.groupby("ticker")["sent_mean"].mean().round(4).sort_values().to_dict()
        if not daily.empty
        else {}
    )

    scatter_df = X.dropna(subset=["sent_mean", "next_ret"])
    scatter_df = scatter_df[scatter_df["n"] > 0]

    headlines = scored.copy()
    if not headlines.empty:
        headlines = headlines.sort_values("ts", ascending=False).head(200)

    topology = {}
    topo_col = "topo_d1_total" if ("topo_d1_total" in price.columns and price["topo_d1_total"].abs().sum() > 0) else "topo_d0_total"
    for t in tickers:
        px = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
        rets = px.pct_change().dropna()
        tp = price[price["ticker"] == t].sort_values("date")
        topology[t] = {
            "returns": [round(float(r), 6) for r in rets.tolist()],
            "dates": [d.isoformat()[:10] for d in tp["date"]],
            "topo": [_clean(v) for v in tp.get(topo_col, pd.Series(dtype=float)).tolist()],
            "topo_label": "H1 loops (persistence total)" if topo_col == "topo_d1_total" else "H0 components (persistence total)",
        }

    return {
        "meta": {
            "mode": req.mode,
            "engine_used": active_engine() or "-",
            "finbert_available": finbert_available(),
            "threshold": req.threshold,
            "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "asof_date": pd.to_datetime(X["date"]).max().isoformat()[:10],
            "last_headline_ts": None if news.empty else pd.to_datetime(news["ts"]).max().isoformat(),
            "n_headlines": int(len(news)),
        },
        "metrics": {k: _clean(v) for k, v in metrics.items()},
        "perf": {
            "date": [d.isoformat()[:10] for d in perf["date"]],
            "equity": [_clean(v) for v in perf["equity"]],
            "bench_equity": [_clean(v) for v in perf["bench_equity"]],
            "drawdown": [_clean(v) for v in perf["drawdown"]],
        },
        "sentiment_by_ticker": sent_by_ticker,
        "scatter": _records(scatter_df, ["ticker", "date", "sent_mean", "next_ret"]),
        "headlines": _records(
            headlines, ["ticker", "ts", "headline", "source", "sent_label", "sent_score"]
        ) if not headlines.empty else [],
        "picks": _records(picks, ["ticker", "p_up", "ret_5", "sent_mean", "n"]) if not picks.empty else [],
        "topology": topology,
    }


# Static mounts (after API routes so /api/* wins)
app.mount("/design", StaticFiles(directory="design"), name="design")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import os

    import uvicorn

    # Cloud platforms (Render, Fly, Railway, HF Spaces) inject $PORT and expect
    # the process to bind 0.0.0.0. Locally this falls back to 127.0.0.1:8000.
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    uvicorn.run(app, host=host, port=port)
