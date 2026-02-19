"""Tests for news fetching module."""
from __future__ import annotations

import pandas as pd
import pytest

from src.news import _normalize_title, _empty_df, _to_utc_datetime
from src.utils import limit_and_sort_headlines


class TestNormalizeTitle:
    """Tests for title normalization."""

    def test_collapses_whitespace(self):
        assert _normalize_title("hello   world") == "hello world"

    def test_strips_leading_trailing(self):
        assert _normalize_title("  hello  ") == "hello"

    def test_handles_none(self):
        assert _normalize_title(None) == ""

    def test_handles_empty(self):
        assert _normalize_title("") == ""

    def test_handles_newlines(self):
        assert _normalize_title("hello\n\nworld") == "hello world"


class TestEmptyDf:
    """Tests for empty DataFrame creation."""

    def test_has_correct_columns(self):
        df = _empty_df()
        expected = ["ticker", "ts", "headline", "source", "link"]
        assert list(df.columns) == expected

    def test_is_empty(self):
        df = _empty_df()
        assert len(df) == 0


class TestToUtcDatetime:
    """Tests for UTC datetime conversion."""

    def test_handles_none(self):
        assert _to_utc_datetime(None) is None

    def test_converts_time_struct(self):
        # feedparser time struct: (year, month, day, hour, min, sec, ...)
        time_struct = (2024, 1, 15, 10, 30, 0, 0, 15, 0)
        result = _to_utc_datetime(time_struct)
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo is not None


class TestLimitAndSortHeadlines:
    """Tests for headline limiting and sorting."""

    def test_limits_count(self):
        df = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=10, freq="h"),
            "headline": [f"headline_{i}" for i in range(10)],
        })
        result = limit_and_sort_headlines(df, max_count=5)
        assert len(result) == 5

    def test_keeps_most_recent(self):
        df = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=10, freq="h"),
            "headline": [f"headline_{i}" for i in range(10)],
        })
        result = limit_and_sort_headlines(df, max_count=3)
        # Should keep last 3 (most recent)
        assert "headline_7" in result["headline"].values
        assert "headline_8" in result["headline"].values
        assert "headline_9" in result["headline"].values

    def test_sorts_chronologically(self):
        df = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=5, freq="h"),
            "headline": [f"headline_{i}" for i in range(5)],
        })
        result = limit_and_sort_headlines(df, max_count=3)
        # Should be in chronological order
        assert result["ts"].is_monotonic_increasing

    def test_handles_empty_df(self):
        df = pd.DataFrame(columns=["ts", "headline"])
        result = limit_and_sort_headlines(df, max_count=5)
        assert len(result) == 0

    def test_handles_none(self):
        result = limit_and_sort_headlines(None, max_count=5)
        assert result is None
