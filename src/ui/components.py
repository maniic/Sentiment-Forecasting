"""
Reusable Streamlit UI components.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import LOCAL_TZ
from src.ui.charts import (
    delay_embedding_fig,
    drawdown_fig,
    equity_fig,
    sentiment_by_ticker_fig,
    sentiment_vs_return_fig,
    topo_timeline_fig,
)

_SENT_BADGES = {"POSITIVE": "🟢 Positive", "NEGATIVE": "🔴 Negative", "NEUTRAL": "⚪ Neutral"}


def format_timestamp(ts: pd.Timestamp | None, tz: str = LOCAL_TZ) -> str:
    """Format a timestamp for display in the local timezone."""
    if ts is None:
        return "-"
    try:
        ts = pd.to_datetime(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(tz).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return str(ts)


def display_freshness_panel(
    latest_headline_ts: pd.Timestamp | None,
    asof_date: pd.Timestamp | None,
    fetched_at: pd.Timestamp | None,
    mode: str,
) -> None:
    """One-line provenance strip: how fresh is what you're looking at."""
    st.caption(
        f"**Last headline:** {format_timestamp(latest_headline_ts)} · "
        f"**As-of date:** {asof_date.strftime('%Y-%m-%d') if asof_date is not None else '-'} · "
        f"**Fetched at:** {format_timestamp(fetched_at)} · "
        f"**Signal engine:** {mode}"
    )


def display_kpi_row(metrics: dict, threshold: float) -> None:
    """Headline numbers for the backtest."""
    ret = metrics.get("total_return")
    bench = metrics.get("bench_total_return")
    delta = None
    if ret is not None and bench is not None and pd.notna(ret) and pd.notna(bench):
        delta = f"{(ret - bench):+.1%} vs buy & hold"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Total return",
        f"{ret:.1%}" if ret is not None and pd.notna(ret) else "—",
        delta=delta,
        help="Strategy return over the backtest window, net of trading costs.",
    )
    c2.metric(
        "Sharpe ratio",
        f"{metrics.get('sharpe', float('nan')):.2f}",
        help="Risk-adjusted return (annualized). Above ~1 is considered good.",
    )
    c3.metric(
        "Max drawdown",
        f"{metrics.get('max_dd', float('nan')):.1%}",
        help="Worst peak-to-trough loss along the way. Closer to 0 is better.",
    )
    win = metrics.get("win_rate")
    c4.metric(
        "Win rate",
        f"{win:.0%}" if win is not None and pd.notna(win) else "—",
        help="Share of invested days that ended positive.",
    )
    c5.metric(
        "Days in market",
        f"{metrics.get('trades', 0)} / {metrics.get('n_days', 0)}",
        help=f"Days holding at least one position (signal threshold {threshold:.2f}).",
    )


def display_performance_tab(perf: pd.DataFrame) -> None:
    """Equity curve + drawdown, both interactive."""
    if perf is None or perf.empty:
        st.info("No performance data — run an analysis first.")
        return
    st.plotly_chart(equity_fig(perf), use_container_width=True)
    st.plotly_chart(drawdown_fig(perf), use_container_width=True)
    st.caption(
        "The blue line is the strategy: each day it holds an equal-weight basket of tickers "
        "whose signal cleared the threshold (or cash if none did), minus trading costs. "
        "The dashed gray line just holds every ticker the whole time."
    )


def display_sentiment_tab(daily: pd.DataFrame, scored: pd.DataFrame, X: pd.DataFrame) -> None:
    """News sentiment visuals + the scored headline feed."""
    if daily is None or daily.empty:
        st.info("No headlines were found for this run.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(sentiment_by_ticker_fig(daily), use_container_width=True)
    with c2:
        if X is not None and not X.empty:
            st.plotly_chart(sentiment_vs_return_fig(X), use_container_width=True)

    if scored is not None and not scored.empty:
        st.markdown("##### Scored headlines")
        view = scored.copy().sort_values("ts", ascending=False)
        view["sentiment"] = view["sent_label"].map(_SENT_BADGES).fillna(view["sent_label"])
        view["when"] = pd.to_datetime(view["ts"]).dt.strftime("%b %d %H:%M")
        st.dataframe(
            view[["ticker", "when", "headline", "sentiment", "sent_score", "source"]],
            use_container_width=True,
            hide_index=True,
            height=380,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "when": st.column_config.TextColumn("When", width="small"),
                "headline": st.column_config.TextColumn("Headline", width="large"),
                "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "sent_score": st.column_config.NumberColumn(
                    "Score", format="%+.2f", width="small",
                    help="Signed confidence: +1 very positive, −1 very negative",
                ),
                "source": st.column_config.TextColumn("Source", width="small"),
            },
        )


def display_todays_picks(picks: pd.DataFrame) -> None:
    """Today's trading signals with probability bars."""
    st.subheader("🎯 Today's Picks")
    if picks is None or picks.empty:
        st.info(
            "No signals today with the current settings — the model isn't confident "
            "enough about any ticker. Try lowering the threshold or disabling the "
            "momentum filter in the sidebar."
        )
        return

    view = picks.copy()
    view["ret_5"] = view["ret_5"] * 100.0  # show momentum as a percentage
    st.dataframe(
        view[["ticker", "p_up", "ret_5", "sent_mean", "n"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "p_up": st.column_config.ProgressColumn(
                "P(up tomorrow)", format="%.2f", min_value=0.0, max_value=1.0,
                help="Model probability that the ticker rises tomorrow",
            ),
            "ret_5": st.column_config.NumberColumn("5-day momentum", format="%+.2f%%"),
            "sent_mean": st.column_config.NumberColumn("News sentiment", format="%+.2f"),
            "n": st.column_config.NumberColumn("# headlines", width="small"),
        },
    )
    st.caption("Educational demo — not investment advice.")


def display_topology_tab(close: pd.DataFrame | None, price_feats: pd.DataFrame, tickers: list[str]) -> None:
    """The Topology Explorer: make the TDA features visible and playable."""
    st.markdown(
        """
        ##### What am I looking at?

        Instead of reading a price chart left-to-right, **topological data analysis (TDA)**
        turns the last few weeks of returns into a *cloud of points* and studies its **shape**.

        - Take a sliding window of daily returns and plot each triple
          *(today, tomorrow, day-after)* as one point in 3D — that's the cloud below.
        - **Persistent homology** then counts the cloud's structure: how many clusters
          (H0) and loops (H1) it has, and how long they persist as you zoom out.
        - Calm, trending markets make tight, simple clouds. Turbulent or cyclical markets
          make stretched clouds with loops. The `topo_*` numbers feed the ML model as
          early-warning "market shape" features that plain returns can't capture.
        """
    )

    if price_feats is None or price_feats.empty:
        st.info("Run an analysis first to populate the topology view.")
        return

    options = [t for t in tickers if t in set(price_feats["ticker"])] or sorted(set(price_feats["ticker"]))
    ticker = st.selectbox("Ticker to explore", options=options, key="topo_ticker")

    if close is not None and ticker in getattr(close, "columns", []):
        returns = close[ticker].pct_change()
        st.plotly_chart(delay_embedding_fig(returns, ticker), use_container_width=True)
        st.caption(
            "Each dot is one day: its position encodes today's, tomorrow's, and the "
            "day-after's return. Color runs light → dark through time. Drag to rotate."
        )

    topo_cols = [c for c in price_feats.columns if c.startswith("topo_")]
    if topo_cols:
        st.plotly_chart(topo_timeline_fig(price_feats, ticker), use_container_width=True)


def display_data_samples(daily: pd.DataFrame, price: pd.DataFrame, n_rows: int = 12) -> None:
    """Sample data tables in an expander."""
    with st.expander("🔍 Peek at the underlying data"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("Daily sentiment (latest rows)")
            st.dataframe(daily.tail(n_rows), use_container_width=True)
        with c2:
            st.write("Price features (latest rows)")
            st.dataframe(price.tail(n_rows), use_container_width=True)


def display_download_buttons(
    news: pd.DataFrame,
    daily: pd.DataFrame,
    price: pd.DataFrame,
    Xp: pd.DataFrame,
    perf: pd.DataFrame,
    metrics: dict,
) -> None:
    """Download buttons for every artifact of the run."""
    with st.expander("⬇️ Download the data from this run"):
        downloadable = {
            "news.csv": news.to_csv(index=False),
            "daily_sentiment.csv": daily.to_csv(index=False),
            "price_features.csv": price.to_csv(index=False),
            "features_joined.csv": Xp.to_csv(index=False),
            "equity_curve.csv": perf.to_csv(index=False),
            "summary_metrics.json": json.dumps(metrics, indent=2),
        }
        cols = st.columns(3)
        for i, (name, data) in enumerate(downloadable.items()):
            with cols[i % 3]:
                st.download_button(
                    label=name, data=data, file_name=name, use_container_width=True
                )


def display_how_it_works() -> None:
    """Plain-English walkthrough of the pipeline."""
    st.markdown(
        """
        ##### The idea

        Markets move on news, but news is text and models need numbers. This app turns
        the day's financial headlines into a trading signal in five steps:

        | Step | What happens | Powered by |
        |------|--------------|------------|
        | 1. 📰 **Collect** | Pull the last few days of headlines for each ticker | Google News RSS / Yahoo Finance |
        | 2. 🧠 **Read** | A finance-tuned AI model reads each headline and scores it positive / negative / neutral | FinBERT (with a lexicon fallback) |
        | 3. 📐 **Shape** | Recent returns are turned into a 3D point cloud and its *shape* is measured — clusters, loops, turbulence | Topological data analysis |
        | 4. 🎲 **Predict** | Sentiment + momentum + shape features → probability each ticker rises tomorrow | Logistic regression / XGBoost / rules |
        | 5. 📊 **Test** | Simulate trading that signal historically, minus costs, against plain buy & hold | Backtester |

        **Demo mode** generates a self-consistent synthetic market (headlines *and* prices,
        offline, reproducible) so you can explore instantly; **Live mode** pulls real
        headlines and real prices.

        > ⚠️ This is an educational project, not investment advice. Backtests on short
        > windows are noisy, and past performance never guarantees anything.
        """
    )
