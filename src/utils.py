"""
Shared utility functions for the Sentiment-Forecasting pipeline.
"""
from __future__ import annotations

import functools
import logging
from typing import Callable, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")


def safe_operation(
    default_return: T | None = None,
    log_errors: bool = True,
    reraise: bool = False,
) -> Callable:
    """
    Decorator for error handling in pipeline operations.

    Parameters
    ----------
    default_return : T | None
        Value to return if an exception occurs (default: None)
    log_errors : bool
        Whether to log exceptions (default: True)
    reraise : bool
        Whether to re-raise exceptions after logging (default: False)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.exception("Error in %s: %s", func.__name__, e)
                if reraise:
                    raise
                return default_return

        return wrapper

    return decorator


def ensure_datetime_column(df: pd.DataFrame, col: str, utc: bool = True) -> pd.DataFrame:
    """
    Ensure a column is datetime type, optionally UTC-aware.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    col : str
        Column name to convert
    utc : bool
        Whether to ensure UTC timezone awareness

    Returns
    -------
    pd.DataFrame
        DataFrame with converted column
    """
    if df is None or df.empty or col not in df.columns:
        return df

    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")

    if utc and out[col].dt.tz is None:
        out[col] = out[col].dt.tz_localize("UTC")
    elif utc and str(out[col].dt.tz) != "UTC":
        out[col] = out[col].dt.tz_convert("UTC")

    return out


def empty_dataframe(columns: list[str]) -> pd.DataFrame:
    """Create an empty DataFrame with specified columns."""
    return pd.DataFrame(columns=columns)


def limit_and_sort_headlines(df: pd.DataFrame, max_count: int, ts_col: str = "ts") -> pd.DataFrame:
    """
    Keep most recent N headlines, then sort chronologically.

    This is a common pattern in both Google and yfinance headline fetchers.

    Parameters
    ----------
    df : pd.DataFrame
        Headlines DataFrame with timestamp column
    max_count : int
        Maximum number of headlines to keep
    ts_col : str
        Name of timestamp column (default: "ts")

    Returns
    -------
    pd.DataFrame
        Filtered and sorted DataFrame
    """
    if df is None or df.empty or max_count <= 0:
        return df

    return (
        df.sort_values(ts_col, ascending=False)
        .head(max_count)
        .sort_values(ts_col)
        .reset_index(drop=True)
    )


def clip_to_range(value: float, min_val: float, max_val: float) -> float:
    """Clip a value to a range."""
    return max(min_val, min(max_val, value))


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"
