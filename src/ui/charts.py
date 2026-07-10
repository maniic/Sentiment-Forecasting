"""
Interactive Plotly figures for the Streamlit app.

All charts share one small design system: a validated categorical palette,
thin marks, recessive grid, direct labels where possible, and unified hover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Palette (validated for colorblind safety and contrast)
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"      # primary series (strategy, positive pole)
VIOLET = "#4a3aa7"    # topology series
RED = "#d03b3b"       # drawdown, negative pole
GRAY = "#898781"      # benchmark / context (always dashed + direct-labeled)
GRID = "#e1e0d9"
INK = "#0b0b0b"
INK_2 = "#52514e"

_BASE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK, size=13),
    margin=dict(l=10, r=10, t=90, b=10),
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(color=INK_2)),
    yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(color=INK_2)),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=12)),
)


def _layout(fig: go.Figure, title: str, **overrides) -> go.Figure:
    merged = {
        **_BASE_LAYOUT,
        "title": dict(text=title, font=dict(size=15), y=0.98, yanchor="top"),
        **overrides,
    }
    fig.update_layout(**merged)
    return fig


def equity_fig(perf: pd.DataFrame) -> go.Figure:
    """Strategy vs buy-and-hold equity curves (growth of $1)."""
    fig = go.Figure()
    if "bench_equity" in perf.columns:
        fig.add_trace(
            go.Scatter(
                x=perf["date"], y=perf["bench_equity"],
                name="Buy & hold (all tickers)",
                line=dict(color=GRAY, width=2, dash="dash"),
                hovertemplate="$%{y:.3f}<extra>Buy & hold</extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=perf["date"], y=perf["equity"],
            name="Sentiment strategy",
            line=dict(color=BLUE, width=2.5),
            hovertemplate="$%{y:.3f}<extra>Strategy</extra>",
        )
    )
    # Direct labels at the line ends
    if len(perf):
        last = perf.iloc[-1]
        fig.add_annotation(
            x=last["date"], y=last["equity"], text=f"<b>Strategy {last['equity']:.2f}x</b>",
            showarrow=False, xanchor="left", xshift=6, font=dict(color=BLUE, size=12),
        )
        if "bench_equity" in perf.columns:
            fig.add_annotation(
                x=last["date"], y=last["bench_equity"], text=f"Buy & hold {last['bench_equity']:.2f}x",
                showarrow=False, xanchor="left", xshift=6, font=dict(color=GRAY, size=12),
            )
    _layout(fig, "Growth of $1 — strategy vs. buy & hold")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(margin=dict(r=110))
    return fig


def drawdown_fig(perf: pd.DataFrame) -> go.Figure:
    """Strategy drawdown from peak (how painful the dips were)."""
    fig = go.Figure(
        go.Scatter(
            x=perf["date"], y=perf["drawdown"],
            name="Drawdown",
            line=dict(color=RED, width=2),
            fill="tozeroy", fillcolor="rgba(208,59,59,0.12)",
            hovertemplate="%{y:.1%}<extra>Drawdown</extra>",
        )
    )
    _layout(fig, "Drawdown from peak", showlegend=False)
    fig.update_yaxes(tickformat=".0%")
    return fig


def sentiment_by_ticker_fig(daily: pd.DataFrame) -> go.Figure:
    """Average news sentiment per ticker (polarity: blue = positive, red = negative)."""
    agg = (
        daily.groupby("ticker")["sent_mean"].mean().sort_values()
        if not daily.empty
        else pd.Series(dtype=float)
    )
    colors = [BLUE if v >= 0 else RED for v in agg.values]
    fig = go.Figure(
        go.Bar(
            x=agg.values, y=agg.index, orientation="h",
            marker=dict(color=colors, cornerradius=4),
            text=[f"{v:+.2f}" for v in agg.values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
            width=0.55,
        )
    )
    _layout(fig, "Average headline sentiment by ticker", showlegend=False, hovermode="closest")
    fig.add_vline(x=0, line_color=GRID, line_width=1)
    fig.update_xaxes(range=[min(-0.05, agg.min() * 1.6) if len(agg) else -1,
                           max(0.05, agg.max() * 1.6) if len(agg) else 1])
    return fig


def sentiment_vs_return_fig(X: pd.DataFrame) -> go.Figure:
    """The core hypothesis in one picture: today's sentiment vs. tomorrow's return."""
    df = X.dropna(subset=["sent_mean", "next_ret"])
    df = df[df["n"] > 0]  # only days that actually had headlines
    fig = go.Figure(
        go.Scatter(
            x=df["sent_mean"], y=df["next_ret"], mode="markers",
            marker=dict(color=BLUE, size=9, opacity=0.55,
                        line=dict(color="#fcfcfb", width=1)),
            customdata=np.stack([df["ticker"], df["date"].astype(str)], axis=-1) if len(df) else None,
            hovertemplate="%{customdata[0]} %{customdata[1]}<br>sentiment %{x:.2f} → next day %{y:.2%}<extra></extra>",
            name="ticker-day",
        )
    )
    # Least-squares trend line
    if len(df) >= 3:
        coeffs = np.polyfit(df["sent_mean"], df["next_ret"], 1)
        xs = np.linspace(df["sent_mean"].min(), df["sent_mean"].max(), 20)
        fig.add_trace(
            go.Scatter(
                x=xs, y=np.polyval(coeffs, xs), mode="lines",
                line=dict(color=INK_2, width=2, dash="dot"),
                name=f"trend (slope {coeffs[0]:+.3f})",
                hoverinfo="skip",
            )
        )
    _layout(fig, "Does today's news predict tomorrow's move?", hovermode="closest")
    fig.update_xaxes(title_text="Headline sentiment (−1 to +1)", title_font=dict(size=12, color=INK_2))
    fig.update_yaxes(title_text="Next-day return", tickformat=".1%", title_font=dict(size=12, color=INK_2))
    return fig


def delay_embedding_fig(returns: pd.Series, ticker: str) -> go.Figure:
    """
    3D view of the delay embedding: each point is (r_t, r_{t+1}, r_{t+2}).

    This is the exact point cloud that persistent homology analyzes — loops
    and clusters in this cloud are what the topo_* features measure.
    """
    r = returns.dropna().values
    if len(r) < 3:
        return go.Figure()
    pts = np.stack([r[:-2], r[1:-1], r[2:]], axis=1)
    t = np.arange(len(pts))
    fig = go.Figure(
        go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers+lines",
            marker=dict(
                size=3.5,
                color=t,
                colorscale=[[0, "#cde2fb"], [1, "#0d366b"]],
                colorbar=dict(title="time →", thickness=12, tickvals=[]),
            ),
            line=dict(color="rgba(42,120,214,0.25)", width=2),
            hovertemplate="r(t)=%{x:.2%}<br>r(t+1)=%{y:.2%}<br>r(t+2)=%{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12),
        title=dict(text=f"{ticker} — daily returns as a 3D point cloud (delay embedding)", font=dict(size=15)),
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="return (t)", yaxis_title="return (t+1)", zaxis_title="return (t+2)",
            xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID), zaxis=dict(gridcolor=GRID),
        ),
        scene_camera=dict(eye=dict(x=0.9, y=0.9, z=0.5)),
        height=520,
    )
    return fig


def topo_timeline_fig(price_feats: pd.DataFrame, ticker: str) -> go.Figure:
    """Topological 'shape complexity' of the market over time for one ticker."""
    df = price_feats[price_feats["ticker"] == ticker].sort_values("date")
    col = "topo_d0_total" if (df.get("topo_d1_total") is None or (df["topo_d1_total"] == 0).all()) else "topo_d1_total"
    label = "H1 loops (persistence total)" if col == "topo_d1_total" else "H0 components (persistence total)"
    fig = go.Figure(
        go.Scatter(
            x=df["date"], y=df[col],
            name=label,
            line=dict(color=VIOLET, width=2),
            fill="tozeroy", fillcolor="rgba(74,58,167,0.10)",
            hovertemplate="%{y:.4f}<extra>" + label + "</extra>",
        )
    )
    _layout(fig, f"{ticker} — topological signal over time", showlegend=False)
    fig.add_annotation(
        xref="paper", yref="paper", x=0, y=1.08, showarrow=False,
        text=f"<span style='color:{INK_2}'>{label}: spikes = the return cloud's shape is changing (regime shifts, turbulence)</span>",
        font=dict(size=12), xanchor="left",
    )
    return fig
