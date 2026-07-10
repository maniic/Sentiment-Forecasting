"""
Streamlit web application for Sentiment Forecasting.

Interactive GUI for analyzing stock sentiment, running backtests,
and viewing trading signals. Ships with an instant offline demo mode
and a live mode that pulls real headlines and prices.
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
)
from src.demo import generate_demo_data
from src.news import fetch_all_headlines
from src.sentiment import (
    active_engine,
    aggregate_daily_sentiment,
    finbert_available,
    score_headlines,
)
from src.features import (
    compute_price_features_from_closes,
    join_sentiment_with_prices,
    load_close_prices,
)
from src.ml import add_quick_prob, load_model, add_ml_prob, DEFAULT_FEATURES
from src.backtest import apply_signals, backtest_equal_weight, todays_signals
from src.ui.components import (
    display_freshness_panel,
    display_kpi_row,
    display_performance_tab,
    display_sentiment_tab,
    display_topology_tab,
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
    "AI reads the day's financial headlines → topology measures the market's shape → "
    "a model turns both into trading signals, backtested against buy & hold. "
    "(Educational demo — not investment advice.)"
)


# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Run Settings")

    data_mode = st.radio(
        "Data source",
        options=["🚀 Demo (instant, offline)", "🌐 Live market data"],
        index=0,
        help="Demo mode simulates a reproducible market with headlines whose tone "
        "genuinely predicts next-day moves — perfect for exploring. Live mode "
        "pulls real headlines and prices from the internet.",
    )
    demo_mode = data_mode.startswith("🚀")

    preset = st.selectbox(
        "Ticker preset",
        options=[
            "Mega Tech (AAPL, MSFT, NVDA, AMZN, GOOGL)",
            "Top ETFs (SPY, QQQ)",
            "Custom",
        ],
    )
    if preset.startswith("Mega Tech"):
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
        st.write("**Tickers:** " + ", ".join(tickers))
    elif preset.startswith("Top ETFs"):
        tickers = ["SPY", "QQQ"]
        st.write("**Tickers:** " + ", ".join(tickers))
    else:
        tickers = st.multiselect("Choose tickers", options=TICKERS, default=TICKERS[:5])

    threshold = st.slider(
        "Signal strength threshold",
        0.45,
        0.75,
        float(PREDICTION_THRESHOLD),
        0.01,
        help="The model only buys when P(up tomorrow) clears this bar. "
        "Higher = fewer but stronger signals.",
    )
    require_mom = st.checkbox(
        "Require positive 5-day momentum",
        value=True,
        help="Extra safety filter: only buy when the stock is already trending up.",
    )

    with st.expander("Advanced settings"):
        news_days = st.slider("Headlines from last N days", 1, 14, DEFAULT_LOOKBACK_DAYS)
        price_lookback = st.slider("Price history (days)", 60, 365, 180)
        cost_bps = st.number_input(
            "Trading cost (bps, round-trip)", value=float(ROUND_TRIP_COST_BPS), step=1.0
        )

        if finbert_available():
            engine_label = st.selectbox(
                "Sentiment engine",
                options=["FinBERT (transformer, most accurate)", "Lexicon (instant, no downloads)"],
                index=1 if demo_mode else 0,
                help="FinBERT downloads a ~400MB model on first use.",
            )
            engine = "finbert" if engine_label.startswith("FinBERT") else "lexicon"
        else:
            st.info("FinBERT not installed — using the built-in lexicon engine.")
            engine = "lexicon"

        if not demo_mode:
            news_source = st.selectbox(
                "News source",
                options=["auto", "google", "yfinance"],
                index=["auto", "google", "yfinance"].index(NEWS_SOURCE),
            )
        else:
            news_source = NEWS_SOURCE

        use_ml = st.checkbox(
            "Use trained ML model (logistic)",
            value=False,
            help="Load a model trained via the Jupyter notebook. Otherwise a "
            "transparent rule blend of sentiment + momentum is used.",
        )
        model_path = st.text_input("Model file", value="models/model_lr.joblib")
        model_meta = st.text_input(
            "Model meta (features/threshold)", value="models/model_lr.meta.json"
        )

    st.markdown("---")
    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)
    refresh = st.button(
        "🔄 Reset / clear results",
        help="Clear cached results; the next run pulls everything fresh.",
        use_container_width=True,
    )


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
        try:
            model = load_model(model_path)
            Xp = add_ml_prob(X, model, feats, out_col="p_up")
            mode = f"ML (Logistic) · {len(feats)} features"
            return Xp, th, mode
        except Exception as e:
            st.warning(f"Could not load model ({e}); using the rule-based blend instead.")
    Xp = add_quick_prob(X)
    return Xp, threshold_fallback, "Rule blend (sentiment + momentum)"


def fetch_and_prepare(_tickers, _news_days, _news_source, _price_lookback, _demo, _engine):
    """Fetch (or simulate) headlines and prices, then build the feature table."""
    if _demo:
        news, close = generate_demo_data(
            _tickers, news_days=_news_days, price_lookback_days=_price_lookback
        )
    else:
        news = fetch_all_headlines(_tickers, _news_days, source=_news_source)
        close = load_close_prices(_tickers, lookback_days=_price_lookback)

    if not news.empty:
        news["ts"] = pd.to_datetime(news["ts"], utc=True, errors="coerce")

    scored = score_headlines(news, engine=_engine)
    daily = aggregate_daily_sentiment(scored)
    price = compute_price_features_from_closes(close)
    X = join_sentiment_with_prices(daily, price)

    return news, scored, daily, price, close, X, _fresh_now_utc()


def run_analysis() -> None:
    """Execute the full pipeline and store results in session state."""
    if not tickers:
        st.warning("Please select at least one ticker.")
        return

    label = "Simulating a market and scoring headlines..." if demo_mode else \
        "Fetching headlines, scoring sentiment, and preparing features..."
    with st.spinner(label):
        news, scored, daily, price, close, X, fetched_at = fetch_and_prepare(
            tickers, int(news_days), news_source, int(price_lookback), demo_mode, engine
        )

    if X.empty:
        st.error(
            "No data came back — if you're in Live mode this usually means the news/price "
            "APIs are unreachable. Try Demo mode to explore offline."
        )
        return

    Xp, th_used, mode = _load_ml_or_quick(X, float(threshold))
    Xs = apply_signals(Xp, threshold=th_used, require_mom_agree=require_mom)
    perf, metrics = backtest_equal_weight(Xs, cost_bps=float(cost_bps))

    engine_used = {"finbert": "FinBERT", "lexicon": "Lexicon"}.get(active_engine(), "—")
    source_tag = "Demo data" if demo_mode else "Live data"

    update_session_state(
        news=news,
        scored=scored,
        daily=daily,
        price=price,
        close=close,
        X=X,
        Xp=Xp,
        perf=perf,
        metrics=metrics,
        th_used=th_used,
        mode=f"{mode} · {engine_used} · {source_tag}",
        fetched_at=fetched_at,
    )


# -------------------- Main content --------------------
if refresh:
    clear_session_state()

if run_btn:
    run_analysis()
elif "metrics" not in st.session_state and demo_mode:
    # First visit: auto-run the demo so the page opens alive, not empty.
    run_analysis()

if "metrics" in st.session_state and st.session_state["metrics"]:
    news = st.session_state["news"]
    scored = st.session_state.get("scored", pd.DataFrame())
    daily = st.session_state["daily"]
    price = st.session_state["price"]
    close = st.session_state.get("close")
    X = st.session_state["X"]
    Xp = st.session_state["Xp"]
    perf = st.session_state["perf"]
    metrics = st.session_state["metrics"]
    th_used = st.session_state["th_used"]
    mode = st.session_state["mode"]

    display_kpi_row(metrics, th_used)

    tab_perf, tab_sent, tab_picks, tab_topo, tab_info = st.tabs(
        [
            "📈 Performance",
            "📰 Headlines & Sentiment",
            "🎯 Today's Picks",
            "🔬 Topology Explorer",
            "ℹ️ How it works",
        ]
    )

    with tab_perf:
        display_performance_tab(perf)

    with tab_sent:
        display_sentiment_tab(daily, scored, X)

    with tab_picks:
        picks = todays_signals(Xp, threshold=th_used, require_mom_agree=require_mom)
        display_todays_picks(picks)

    with tab_topo:
        display_topology_tab(close, price, tickers)

    with tab_info:
        display_how_it_works()

    st.markdown("---")
    latest_head_ts = None if news.empty else pd.to_datetime(news["ts"]).max()
    asof_date = None if X.empty else pd.to_datetime(X["date"]).max()
    display_freshness_panel(
        latest_headline_ts=latest_head_ts,
        asof_date=asof_date,
        fetched_at=st.session_state.get("fetched_at"),
        mode=mode,
    )
    display_data_samples(daily, price)
    display_download_buttons(
        news=news, daily=daily, price=price, Xp=Xp, perf=perf, metrics=metrics
    )
else:
    st.info("👈 Pick your settings in the sidebar and hit **Run Analysis** to get started.")
    display_how_it_works()


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
