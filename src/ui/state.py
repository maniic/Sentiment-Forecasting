"""
Session state management for the Streamlit app.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import streamlit as st


@dataclass
class SessionState:
    """
    Typed container for Streamlit session state.

    Provides a clean interface for managing the data stored
    between Streamlit reruns.
    """

    news: pd.DataFrame = field(default_factory=pd.DataFrame)
    scored: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    price: pd.DataFrame = field(default_factory=pd.DataFrame)
    X: pd.DataFrame = field(default_factory=pd.DataFrame)
    Xp: pd.DataFrame = field(default_factory=pd.DataFrame)
    perf: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)
    th_used: float = 0.5
    mode: str = ""
    fetched_at: pd.Timestamp | None = None

    def is_populated(self) -> bool:
        """Check if results are available."""
        return bool(self.metrics)


# Session state keys
_STATE_KEYS = [
    "news",
    "scored",
    "daily",
    "price",
    "close",
    "X",
    "Xp",
    "perf",
    "metrics",
    "th_used",
    "mode",
    "fetched_at",
]


def get_session_state() -> SessionState:
    """
    Get the current session state as a SessionState object.

    Returns
    -------
    SessionState
        Current session state data
    """
    return SessionState(
        news=st.session_state.get("news", pd.DataFrame()),
        scored=st.session_state.get("scored", pd.DataFrame()),
        daily=st.session_state.get("daily", pd.DataFrame()),
        price=st.session_state.get("price", pd.DataFrame()),
        X=st.session_state.get("X", pd.DataFrame()),
        Xp=st.session_state.get("Xp", pd.DataFrame()),
        perf=st.session_state.get("perf", pd.DataFrame()),
        metrics=st.session_state.get("metrics", {}),
        th_used=st.session_state.get("th_used", 0.5),
        mode=st.session_state.get("mode", ""),
        fetched_at=st.session_state.get("fetched_at"),
    )


def save_session_state(state: SessionState) -> None:
    """
    Save a SessionState object to Streamlit session state.

    Parameters
    ----------
    state : SessionState
        State to save
    """
    st.session_state["news"] = state.news
    st.session_state["scored"] = state.scored
    st.session_state["daily"] = state.daily
    st.session_state["price"] = state.price
    st.session_state["X"] = state.X
    st.session_state["Xp"] = state.Xp
    st.session_state["perf"] = state.perf
    st.session_state["metrics"] = state.metrics
    st.session_state["th_used"] = state.th_used
    st.session_state["mode"] = state.mode
    st.session_state["fetched_at"] = state.fetched_at


def clear_session_state() -> None:
    """Clear all session state data."""
    for key in _STATE_KEYS:
        st.session_state.pop(key, None)


def update_session_state(**kwargs: Any) -> None:
    """
    Update specific session state values.

    Parameters
    ----------
    **kwargs
        Key-value pairs to update
    """
    for key, value in kwargs.items():
        if key in _STATE_KEYS:
            st.session_state[key] = value
