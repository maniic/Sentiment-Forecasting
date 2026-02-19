from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.config import TopoConfig

logger = logging.getLogger(__name__)

# Optional giotto-tda import — if missing, we fallback to lightweight stats
try:
    from gtda.homology import VietorisRipsPersistence
    HAS_GTDA = True
except Exception:  # pragma: no cover - optional dep
    HAS_GTDA = False
    VietorisRipsPersistence = None  # type: ignore


def _delay_embed(series: np.ndarray, embed_dim: int) -> np.ndarray:
    """Simple time-delay embedding: shape (n_windows, embed_dim)."""
    if len(series) < embed_dim:
        return np.empty((0, embed_dim))
    windows = []
    for i in range(len(series) - embed_dim + 1):
        windows.append(series[i : i + embed_dim])
    return np.asarray(windows)


def _summarize_diagram(diag: np.ndarray, dim: int) -> dict[str, float]:
    """Compute persistence summaries for a given homology dimension."""
    if diag.size == 0:
        return {
            f"topo_d{dim}_total": 0.0,
            f"topo_d{dim}_max": 0.0,
            f"topo_d{dim}_mean": 0.0,
            f"topo_d{dim}_count": 0.0,
        }
    filt = diag[diag[:, 2] == dim]
    if filt.size == 0:
        return {
            f"topo_d{dim}_total": 0.0,
            f"topo_d{dim}_max": 0.0,
            f"topo_d{dim}_mean": 0.0,
            f"topo_d{dim}_count": 0.0,
        }
    life = np.clip(filt[:, 1] - filt[:, 0], a_min=0.0, a_max=None)
    total = float(np.sum(life))
    max_life = float(np.max(life))
    mean_life = float(np.mean(life))
    count = float(len(life))
    return {
        f"topo_d{dim}_total": total,
        f"topo_d{dim}_max": max_life,
        f"topo_d{dim}_mean": mean_life,
        f"topo_d{dim}_count": count,
    }


def _fallback_stats(window_vals: np.ndarray) -> dict[str, float]:
    """If giotto-tda is unavailable, emit robust higher-moment stats as a proxy."""
    if window_vals.size == 0:
        return {
            "topo_d0_total": 0.0,
            "topo_d0_max": 0.0,
            "topo_d0_mean": 0.0,
            "topo_d0_count": 0.0,
            "topo_d1_total": 0.0,
            "topo_d1_max": 0.0,
            "topo_d1_mean": 0.0,
            "topo_d1_count": 0.0,
            "topo_entropy": 0.0,
        }
    var = float(np.var(window_vals))
    kurt = float(pd.Series(window_vals).kurtosis(skipna=True))
    entropy = float(pd.Series(window_vals).value_counts(normalize=True).pow(2).sum())
    return {
        "topo_d0_total": var,
        "topo_d0_max": kurt,
        "topo_d0_mean": var,
        "topo_d0_count": float(len(window_vals)),
        "topo_d1_total": 0.0,
        "topo_d1_max": 0.0,
        "topo_d1_mean": 0.0,
        "topo_d1_count": 0.0,
        "topo_entropy": entropy,
    }


def summarize_persistence(window_vals: np.ndarray, cfg: TopoConfig) -> dict[str, float]:
    """
    Compute persistence features on a rolling window of returns/prices.
    Returns a dict of topo features (D0/D1 total/max/mean/count) plus entropy proxy.
    If giotto-tda is unavailable, returns fallback stats to keep pipeline functional.
    """
    window_vals = window_vals.astype(float)
    window_vals = window_vals[~np.isnan(window_vals)]
    if window_vals.size < max(cfg.embed_dim, 5):
        return _fallback_stats(window_vals)

    cloud = _delay_embed(window_vals, cfg.embed_dim)
    if cloud.size == 0:
        return _fallback_stats(window_vals)

    if not HAS_GTDA or VietorisRipsPersistence is None:
        return _fallback_stats(window_vals)

    try:  # pragma: no cover - optional dep path
        vr = VietorisRipsPersistence(
            homology_dimensions=cfg.homology_dims, metric=cfg.metric, collapse_edges=True
        )
        diagrams = vr.fit_transform([cloud])
        diag = diagrams[0]
        feats = {"topo_entropy": float(np.var(cloud))}
        for d in cfg.homology_dims:
            feats.update(_summarize_diagram(diag, d))
        return feats
    except Exception as e:  # pragma: no cover - robustness
        logger.warning("Topology features failed; falling back. err=%s", e)
        return _fallback_stats(window_vals)


def compute_topo_features_for_series(
    px: pd.Series,
    ticker: str,
    cfg: TopoConfig | None = None,
) -> pd.DataFrame:
    """
    Compute per-date topological features for a price/return series.
    Uses returns (pct change) to build delay-embedded point clouds.
    """
    cfg = cfg or TopoConfig()
    ret = px.pct_change().dropna()
    records = []
    for idx in range(cfg.window, len(ret)):
        window_slice = ret.iloc[idx - cfg.window : idx].values
        feats = summarize_persistence(window_slice, cfg)
        records.append({"date": ret.index[idx], "ticker": ticker, **feats})
    if not records:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "topo_d0_total",
                "topo_d0_max",
                "topo_d0_mean",
                "topo_d0_count",
                "topo_d1_total",
                "topo_d1_max",
                "topo_d1_mean",
                "topo_d1_count",
                "topo_entropy",
            ]
        )
    out = pd.DataFrame(records)
    out["date"] = pd.to_datetime(pd.to_datetime(out["date"]).dt.date)
    return out


def merge_topology_features(
    price_df: pd.DataFrame,
    topo_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge topology features into the base price feature frame."""
    if topo_df is None or topo_df.empty:
        return price_df.copy()
    merged = price_df.merge(topo_df, on=["ticker", "date"], how="left")
    topo_cols = [c for c in merged.columns if c.startswith("topo_")]
    merged[topo_cols] = merged[topo_cols].fillna(0.0)
    return merged


def topo_feature_names() -> list[str]:
    """List of topo feature columns produced by this module."""
    return [
        "topo_d0_total",
        "topo_d0_max",
        "topo_d0_mean",
        "topo_d0_count",
        "topo_d1_total",
        "topo_d1_max",
        "topo_d1_mean",
        "topo_d1_count",
        "topo_entropy",
    ]
