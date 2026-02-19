"""
Streamlit web application for Sentiment Forecasting.

Interactive GUI for analyzing stock sentiment, running backtests,
and viewing trading signals.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    TICKERS,
    DEFAULT_LOOKBACK_DAYS,
    PREDICTION_THRESHOLD,
    ROUND_TRIP_COST_BPS,
    NEWS_SOURCE,
    LOCAL_TZ,
)
from src.news import fetch_all_headlines
from src.sentiment import score_headlines, aggregate_daily_sentiment
from src.features import compute_price_features, join_sentiment_with_prices
from src.ml import add_quick_prob, load_model, add_ml_prob, DEFAULT_FEATURES
from src.backtest import apply_signals, backtest_equal_weight, todays_signals
from src.ui.components import (
    display_freshness_panel,
    display_metrics_panel,
    display_equity_chart,
    display_data_samples,
    display_download_buttons,
    display_todays_picks,
    display_how_it_works,
)
from src.ui.state import clear_session_state, update_session_state


# -------------------- Page config --------------------
st.set_page_config(page_title="Sentiment Forecasting", page_icon="📰", layout="wide")
st.title("📰📈 Sentiment Forecasting")
st.caption(
    "FinBERT sentiment on headlines + simple price momentum → signals → backtest. "
    "(Demo only; not investment advice.)"
)


# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Run Settings")

    level = st.radio("Experience level", options=["Beginner", "Advanced"], index=0)
    preset = st.selectbox(
        "Ticker preset",
        options=[
            "Top ETFs (SPY, QQQ)",
            "Mega Tech (AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA)",
            "Custom",
        ],
    )

    if preset == "Top ETFs (SPY, QQQ)":
        default_tickers = ["SPY", "QQQ"]
    elif preset == "Mega Tech (AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA)":
        default_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
    else:
        default_tickers = TICKERS[:5]

    if preset == "Custom":
        tickers = st.multiselect("Choose tickers", options=TICKERS, default=default_tickers)
    else:
        st.write("**Tickers:** " + ", ".join(default_tickers))
        tickers = default_tickers

    if level == "Beginner":
        news_days = st.slider(
            "Headlines from last N days",
            3,
            10,
            max(3, min(7, DEFAULT_LOOKBACK_DAYS)),
        )
        threshold = st.slider(
            "Signal strength threshold",
            0.50,
            0.70,
            max(0.55, float(PREDICTION_THRESHOLD)),
            0.01,
            help="Higher = fewer but stronger signals",
        )
        require_mom = st.checkbox("Require positive momentum (5-day)", value=True)
        price_lookback = 180
        cost_bps = float(ROUND_TRIP_COST_BPS)
        news_source = NEWS_SOURCE
        use_ml = False
        model_path = "output/logreg_sentiment.joblib"
        model_meta = "output/logreg_sentiment.meta.json"
    else:
        news_days = st.slider("Headlines from last N days", 1, 14, DEFAULT_LOOKBACK_DAYS)
        price_lookback = st.slider("Price lookback (days)", 60, 365, 180)
        news_source = st.selectbox(
            "News source",
            options=["auto", "google", "yfinance"],
            index=["auto", "google", "yfinance"].index(NEWS_SOURCE),
        )
        threshold = st.slider(
            "Signal strength threshold",
            0.45,
            0.75,
            float(PREDICTION_THRESHOLD),
            0.01,
        )
        require_mom = st.checkbox("Require positive momentum (5-day)", value=True)
        cost_bps = st.number_input(
            "Trading cost (bps, round-trip)",
            value=float(ROUND_TRIP_COST_BPS),
            step=1.0,
        )
        use_ml = st.checkbox(
            "Use trained ML model (logistic)",
            value=False,
            help="Train a model first using the Jupyter notebook or CLI",
        )
        model_path = st.text_input("Model file", value="output/logreg_sentiment.joblib")
        model_meta = st.text_input(
            "Model meta (features/threshold)",
            value="output/logreg_sentiment.meta.json",
        )

    st.markdown("---")
    refresh = st.button(
        "🔄 Refresh data (force new pull)",
        help="Bypass any cached state and fetch everything fresh.",
    )
    run_btn = st.button("Run Analysis", type="primary")


# -------------------- Helper functions --------------------
def _fresh_now_utc() -> pd.Timestamp:
    """Return a tz-aware UTC timestamp."""
    return pd.Timestamp.now(tz="UTC")


def _load_ml_or_quick(X: pd.DataFrame, threshold_fallback: float):
    """Return (Xp, threshold_used, mode_str)."""
    if use_ml and Path(model_path).exists():
        feats = DEFAULT_FEATURES
        th = threshold_fallback
        try:
            meta = json.loads(Path(model_meta).read_text())
            feats = meta.get("features", feats)
            th = float(meta.get("threshold", th))
        except Exception:
            pass
        model = load_model(model_path)
        Xp = add_ml_prob(X, model, feats, out_col="p_up")
        mode = f"ML (Logistic) · features={len(feats)} · threshold={th:.2f}"
        return Xp, th, mode
    else:
        Xp = add_quick_prob(X)
        mode = "Quick Prob (rule-based)"
        return Xp, threshold_fallback, mode


def fetch_and_prepare(_tickers, _news_days, _news_source, _price_lookback):
    """
    LIVE data path - no Streamlit caching so each run pulls fresh data.
    """
    news = fetch_all_headlines(_tickers, _news_days, source=_news_source)
    if not news.empty:
        news["ts"] = pd.to_datetime(news["ts"], utc=True, errors="coerce")

    scored = score_headlines(news)
    daily = aggregate_daily_sentiment(scored)
    price = compute_price_features(_tickers, lookback_days=_price_lookback)
    X = join_sentiment_with_prices(daily, price)

    fetched_at = _fresh_now_utc()
    return news, scored, daily, price, X, fetched_at


# -------------------- Main content --------------------
display_how_it_works()

# Clear session state if user hits Refresh
if refresh:
    clear_session_state()

if run_btn:
    if not tickers:
        st.warning("Please select at least one ticker.")
    else:
        with st.spinner("Fetching headlines, scoring sentiment, and preparing features..."):
            news, scored, daily, price, X, fetched_at = fetch_and_prepare(
                tickers, int(news_days), news_source, int(price_lookback)
            )

        Xp, th_used, mode = _load_ml_or_quick(X, float(threshold))
        Xs = apply_signals(Xp, threshold=th_used, require_mom_agree=require_mom)
        perf, metrics = backtest_equal_weight(Xs, cost_bps=float(cost_bps))

        update_session_state(
            news=news,
            daily=daily,
            price=price,
            X=X,
            Xp=Xp,
            perf=perf,
            metrics=metrics,
            th_used=th_used,
            mode=mode,
            fetched_at=fetched_at,
        )
        st.success("Run complete.")

# -------------------- Results display --------------------
if "metrics" in st.session_state and st.session_state["metrics"]:
    news = st.session_state["news"]
    X = st.session_state["X"]
    Xp = st.session_state["Xp"]
    perf = st.session_state["perf"]
    metrics = st.session_state["metrics"]
    th_used = st.session_state["th_used"]
    mode = st.session_state["mode"]

    # Freshness panel
    latest_head_ts = None if news.empty else pd.to_datetime(news["ts"]).max()
    asof_date = None if X.empty else pd.to_datetime(X["date"]).max()

    st.markdown("### Results")
    with st.container():
        display_freshness_panel(
            latest_headline_ts=latest_head_ts,
            asof_date=asof_date,
            fetched_at=st.session_state.get("fetched_at"),
            mode=mode,
        )

    # Main results
    c_top = st.columns([2, 1])
    with c_top[0]:
        display_equity_chart(perf)
    with c_top[1]:
        display_metrics_panel(metrics, th_used)

    # Today's picks
    picks = todays_signals(Xp, threshold=th_used, require_mom_agree=require_mom)
    display_todays_picks(picks)

    # Data samples
    display_data_samples(st.session_state["daily"], st.session_state["price"])

    # Downloads
    display_download_buttons(
        news=news,
        daily=st.session_state["daily"],
        price=st.session_state["price"],
        Xp=Xp,
        perf=perf,
        metrics=metrics,
    )


# -------------------- Guard --------------------
if __name__ == "__main__":
    import sys

    print(
        "\nThis is a Streamlit app. Launch it with:\n"
        "  streamlit run app.py\n"
        "  python -m streamlit run app.py\n"
        "If port 8501 is busy: streamlit run app.py --server.port 8502\n"
    )
    sys.exit(0)
