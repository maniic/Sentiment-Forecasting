# app.py
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.config import (
    TICKERS, DEFAULT_LOOKBACK_DAYS, PREDICTION_THRESHOLD, ROUND_TRIP_COST_BPS, NEWS_SOURCE
)
from src.news import fetch_all_headlines
from src.sentiment import score_headlines, aggregate_daily_sentiment
from src.features import compute_price_features, join_sentiment_with_prices
from src.ml import add_quick_prob
from src.backtest import apply_signals, backtest_equal_weight, todays_signals
from src.visuals import plot_equity


# -------------------- Page config --------------------
st.set_page_config(page_title="Sentiment Forecasting", page_icon="📰", layout="wide")
st.title("📰📈 Sentiment Forecasting")
st.caption("FinBERT sentiment on headlines + simple price momentum → signals → backtest. (Demo only; not investment advice.)")

LOCAL_TZ = "America/Toronto"  # display timestamps in your local zone


# -------------------- Sidebar: simple + advanced --------------------
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
        news_days = st.slider("Headlines from last N days", 3, 10, max(3, min(7, DEFAULT_LOOKBACK_DAYS)))
        threshold = st.slider("Signal strength threshold", 0.50, 0.70, max(0.55, float(PREDICTION_THRESHOLD)), 0.01,
                              help="Higher = fewer but stronger signals")
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
        news_source = st.selectbox("News source", options=["auto", "google", "yfinance"],
                                   index=["auto","google","yfinance"].index(NEWS_SOURCE))
        threshold = st.slider("Signal strength threshold", 0.45, 0.75, float(PREDICTION_THRESHOLD), 0.01)
        require_mom = st.checkbox("Require positive momentum (5-day)", value=True)
        cost_bps = st.number_input("Trading cost (bps, round-trip)", value=float(ROUND_TRIP_COST_BPS), step=1.0)
        use_ml = st.checkbox("Use trained ML model (logistic)", value=False,
                             help="Train with scratch_test_ml.py (saves model in /output)")
        model_path = st.text_input("Model file", value="output/logreg_sentiment.joblib")
        model_meta = st.text_input("Model meta (features/threshold)", value="output/logreg_sentiment.meta.json")

    st.markdown("---")
    refresh = st.button("🔄 Refresh data (force new pull)", help="Bypass any cached state and fetch everything fresh.")
    run_btn = st.button("Run Analysis", type="primary")


# -------------------- Helpers --------------------
def _fresh_now_utc():
    # Always return a tz-aware UTC timestamp
    return pd.Timestamp.now(tz="UTC")

def _fmt_ts(ts):
    if ts is None:
        return "—"
    try:
        ts = pd.to_datetime(ts)
        # If naive → localize to UTC; if already tz-aware → leave as is
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return str(ts)
    
def _load_ml_or_quick(X: pd.DataFrame, threshold_fallback: float):
    """Return (Xp, threshold_used, mode_str)."""
    if use_ml and Path(model_path).exists():
        from src.ml import load_model, add_ml_prob, DEFAULT_FEATURES
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
        mode = f"ML (Logistic) · features={feats} · threshold={th:.2f}"
        return Xp, th, mode
    else:
        Xp = add_quick_prob(X)
        mode = "Quick Prob (rule-based)"
        return Xp, threshold_fallback, mode

def fetch_and_prepare(_tickers, _news_days, _news_source, _price_lookback):
    """
    LIVE data path — no Streamlit caching on purpose so each run pulls fresh headlines/prices.
    """
    news = fetch_all_headlines(_tickers, _news_days, source=_news_source)
    # ensure tz-aware
    if not news.empty:
        news["ts"] = pd.to_datetime(news["ts"], utc=True, errors="coerce")

    scored = score_headlines(news)
    daily = aggregate_daily_sentiment(scored)
    price = compute_price_features(_tickers, lookback_days=_price_lookback)
    X = join_sentiment_with_prices(daily, price)

    fetched_at = _fresh_now_utc()
    return news, scored, daily, price, X, fetched_at


# -------------------- Main content --------------------
st.markdown("### How it works")
st.markdown(
    "1) Pull recent headlines → 2) FinBERT sentiment → 3) Add price momentum → "
    "4) Compute probability of up move → 5) Threshold → 6) Backtest (equal-weight, costs)."
)

# Clear session state if user hits Refresh
if refresh:
    for k in ("news","scored","daily","price","X","Xp","perf","metrics","th_used","mode","fetched_at"):
        st.session_state.pop(k, None)

if run_btn:
    if not tickers:
        st.warning("Please select at least one ticker.")
    else:
        with st.spinner("Fetching headlines, scoring sentiment, and preparing features…"):
            news, scored, daily, price, X, fetched_at = fetch_and_prepare(
                tickers, int(news_days), news_source, int(price_lookback)
            )

        Xp, th_used, mode = _load_ml_or_quick(X, float(threshold))
        Xs = apply_signals(Xp, threshold=th_used, require_mom_agree=require_mom)
        perf, metrics = backtest_equal_weight(Xs, cost_bps=float(cost_bps))

        st.session_state["news"] = news
        st.session_state["daily"] = daily
        st.session_state["price"] = price
        st.session_state["X"] = X
        st.session_state["Xp"] = Xp
        st.session_state["perf"] = perf
        st.session_state["metrics"] = metrics
        st.session_state["th_used"] = th_used
        st.session_state["mode"] = mode
        st.session_state["fetched_at"] = fetched_at
        st.success("Run complete.")

# Results
if "metrics" in st.session_state:
    # Freshness panel
    news = st.session_state["news"]
    latest_head_ts = None if news.empty else pd.to_datetime(news["ts"]).max()
    asof_date = None if st.session_state["X"].empty else pd.to_datetime(st.session_state["X"]["date"]).max()

    st.markdown("### Results")
    with st.container():
        c0, c1, c2, c3 = st.columns([1.2, 1, 1, 1])
        c0.metric("Last headline time", _fmt_ts(latest_head_ts) if latest_head_ts is not None else "—")
        c1.metric("As-of (features date)", asof_date.strftime("%Y-%m-%d") if asof_date is not None else "—")
        c2.metric("Fetched at", _fmt_ts(st.session_state.get("fetched_at")))
        c3.metric("Mode", st.session_state["mode"])

    c_top = st.columns([2, 1])
    with c_top[0]:
        st.subheader("Equity Curve")
        eq_path = Path("output/equity_curve_gui.png")
        eq_path.parent.mkdir(parents=True, exist_ok=True)
        plot_equity(st.session_state["perf"], title="Equity Curve", save_path=str(eq_path))
        st.image(str(eq_path), use_column_width=True, caption="Cumulative equity (net of simple costs)")

    with c_top[1]:
        st.subheader("Summary")
        m = st.session_state["metrics"]
        c1, c2 = st.columns(2)
        c1.metric("Sharpe (net)", f"{m.get('sharpe', float('nan')):.2f}")
        c2.metric("Max Drawdown", f"{m.get('max_dd', float('nan')):.2%}")
        c1.metric("Trade Days", f"{m.get('trades', 0)}")
        c2.metric("Threshold", f"{st.session_state['th_used']:.2f}")

    st.subheader("Today’s Picks")
    picks = todays_signals(st.session_state["Xp"], threshold=st.session_state["th_used"], require_mom_agree=require_mom)
    if picks.empty:
        st.info("No signals today with the current settings.")
    else:
        st.dataframe(picks, use_container_width=True)

    with st.expander("Data samples (click to view)"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("Daily sentiment (tail)")
            st.dataframe(st.session_state["daily"].tail(12), use_container_width=True)
        with c2:
            st.write("Price features (tail)")
            st.dataframe(st.session_state["price"].tail(12), use_container_width=True)

    st.subheader("Downloads")
    outdir = Path("output"); outdir.mkdir(parents=True, exist_ok=True)
    downloadable = {
        "news.csv": st.session_state["news"].to_csv(index=False),
        "daily_sentiment.csv": st.session_state["daily"].to_csv(index=False),
        "price_features.csv": st.session_state["price"].to_csv(index=False),
        "features_joined.csv": st.session_state["Xp"].to_csv(index=False),
        "equity_curve.csv": st.session_state["perf"].to_csv(index=False),
        "summary_metrics.json": json.dumps(st.session_state["metrics"], indent=2),
    }
    cols = st.columns(3)
    i = 0
    for name, data in downloadable.items():
        with cols[i % 3]:
            st.download_button(label=f"Download {name}", data=data, file_name=name, use_container_width=True)
        i += 1


# -------------------- Friendly guard (if someone runs `python app.py`) --------------------
if __name__ == "__main__":
    import sys
    print(
        "\nThis is a Streamlit app. Launch it with one:\n"
        "  streamlit run app.py\n"
        "  python -m streamlit run app.py\n"
        "If port 8501 is busy: streamlit run app.py --server.port 8502\n"
    )
    sys.exit(0)
