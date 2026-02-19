"""
Reusable Streamlit UI components.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import LOCAL_TZ, OUTPUT_DIR
from src.visuals import plot_equity


def format_timestamp(ts: pd.Timestamp | None, tz: str = LOCAL_TZ) -> str:
    """
    Format a timestamp for display.

    Parameters
    ----------
    ts : pd.Timestamp | None
        Timestamp to format
    tz : str
        Target timezone for display

    Returns
    -------
    str
        Formatted timestamp string
    """
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
    """
    Display the data freshness panel.

    Parameters
    ----------
    latest_headline_ts : pd.Timestamp | None
        Timestamp of most recent headline
    asof_date : pd.Timestamp | None
        Most recent feature date
    fetched_at : pd.Timestamp | None
        When data was fetched
    mode : str
        Probability computation mode
    """
    c0, c1, c2, c3 = st.columns([1.2, 1, 1, 1])
    c0.metric("Last headline time", format_timestamp(latest_headline_ts))
    c1.metric(
        "As-of (features date)",
        asof_date.strftime("%Y-%m-%d") if asof_date is not None else "-",
    )
    c2.metric("Fetched at", format_timestamp(fetched_at))
    c3.metric("Mode", mode)


def display_metrics_panel(metrics: dict, threshold: float) -> None:
    """
    Display the summary metrics panel.

    Parameters
    ----------
    metrics : dict
        Backtest metrics dictionary
    threshold : float
        Probability threshold used
    """
    st.subheader("Summary")
    c1, c2 = st.columns(2)
    c1.metric("Sharpe (net)", f"{metrics.get('sharpe', float('nan')):.2f}")
    c2.metric("Max Drawdown", f"{metrics.get('max_dd', float('nan')):.2%}")
    c1.metric("Trade Days", f"{metrics.get('trades', 0)}")
    c2.metric("Threshold", f"{threshold:.2f}")


def display_equity_chart(
    perf: pd.DataFrame,
    title: str = "Equity Curve",
    save_path: str | Path | None = None,
) -> None:
    """
    Display the equity curve chart.

    Parameters
    ----------
    perf : pd.DataFrame
        Performance DataFrame with date and equity columns
    title : str
        Chart title
    save_path : str | Path | None
        Path to save the chart image
    """
    st.subheader("Equity Curve")

    if save_path is None:
        save_path = OUTPUT_DIR / "equity_curve_gui.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plot_equity(perf, title=title, save_path=str(save_path))
    st.image(str(save_path), use_column_width=True, caption="Cumulative equity (net of simple costs)")


def display_data_samples(daily: pd.DataFrame, price: pd.DataFrame, n_rows: int = 12) -> None:
    """
    Display sample data tables in an expander.

    Parameters
    ----------
    daily : pd.DataFrame
        Daily sentiment DataFrame
    price : pd.DataFrame
        Price features DataFrame
    n_rows : int
        Number of rows to display
    """
    with st.expander("Data samples (click to view)"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("Daily sentiment (tail)")
            st.dataframe(daily.tail(n_rows), use_container_width=True)
        with c2:
            st.write("Price features (tail)")
            st.dataframe(price.tail(n_rows), use_container_width=True)


def display_download_buttons(
    news: pd.DataFrame,
    daily: pd.DataFrame,
    price: pd.DataFrame,
    Xp: pd.DataFrame,
    perf: pd.DataFrame,
    metrics: dict,
) -> None:
    """
    Display download buttons for all data files.

    Parameters
    ----------
    news : pd.DataFrame
        Raw news headlines
    daily : pd.DataFrame
        Daily sentiment
    price : pd.DataFrame
        Price features
    Xp : pd.DataFrame
        Joined features with probabilities
    perf : pd.DataFrame
        Performance data
    metrics : dict
        Summary metrics
    """
    st.subheader("Downloads")

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
                label=f"Download {name}",
                data=data,
                file_name=name,
                use_container_width=True,
            )


def display_todays_picks(picks: pd.DataFrame) -> None:
    """
    Display today's trading signals.

    Parameters
    ----------
    picks : pd.DataFrame
        DataFrame of today's signals
    """
    st.subheader("Today's Picks")
    if picks.empty:
        st.info("No signals today with the current settings.")
    else:
        st.dataframe(picks, use_container_width=True)


def display_how_it_works() -> None:
    """Display the 'How it works' section."""
    st.markdown("### How it works")
    st.markdown(
        "1) Pull recent headlines -> 2) FinBERT sentiment -> 3) Add price momentum -> "
        "4) Compute probability of up move -> 5) Threshold -> 6) Backtest (equal-weight, costs)."
    )
