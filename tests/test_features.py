"""Tests for feature engineering module."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.features import (
    _as_list,
    _add_labels,
    _empty_price_features,
    join_sentiment_with_prices,
)
from src.schemas import Columns


class TestAsList:
    """Tests for ticker list conversion."""

    def test_string_to_list(self):
        assert _as_list("AAPL") == ["AAPL"]

    def test_list_unchanged(self):
        assert _as_list(["AAPL", "MSFT"]) == ["AAPL", "MSFT"]

    def test_tuple_to_list(self):
        assert _as_list(("AAPL", "MSFT")) == ["AAPL", "MSFT"]


class TestAddLabels:
    """Tests for label generation."""

    def test_positive_return_label_1(self):
        df = pd.DataFrame({"next_ret": [0.01, 0.02]})
        result = _add_labels(df)
        assert all(result["label"] == 1)

    def test_negative_return_label_0(self):
        df = pd.DataFrame({"next_ret": [-0.01, -0.02]})
        result = _add_labels(df)
        assert all(result["label"] == 0)

    def test_zero_return_label_0(self):
        df = pd.DataFrame({"next_ret": [0.0]})
        result = _add_labels(df)
        assert result["label"].iloc[0] == 0

    def test_mixed_returns(self):
        df = pd.DataFrame({"next_ret": [0.01, -0.01, 0.0, 0.02]})
        result = _add_labels(df)
        expected = [1, 0, 0, 1]
        assert list(result["label"]) == expected


class TestEmptyPriceFeatures:
    """Tests for empty price features DataFrame."""

    def test_has_correct_columns(self):
        df = _empty_price_features()
        expected_cols = [
            "ticker", "date", "ret_1", "ret_5", "ret_10",
            "ret_20", "vol_20", "next_ret", "label"
        ]
        assert list(df.columns) == expected_cols

    def test_is_empty(self):
        df = _empty_price_features()
        assert len(df) == 0


class TestJoinSentimentWithPrices:
    """Tests for joining sentiment with price features."""

    def test_joins_on_ticker_date(self, sample_price_features, sample_daily_sentiment):
        result = join_sentiment_with_prices(sample_daily_sentiment, sample_price_features)

        # Should have sent_mean column
        assert "sent_mean" in result.columns
        assert "sent_std" in result.columns
        assert "n" in result.columns

    def test_fills_missing_sentiment_with_zero(self, sample_price_features):
        # Empty sentiment
        empty_sent = pd.DataFrame(columns=["ticker", "date", "sent_mean", "sent_std", "n"])
        result = join_sentiment_with_prices(empty_sent, sample_price_features)

        assert (result["sent_mean"] == 0.0).all()
        assert (result["sent_std"] == 0.0).all()

    def test_handles_empty_price_features(self, sample_daily_sentiment):
        empty_price = pd.DataFrame()
        result = join_sentiment_with_prices(sample_daily_sentiment, empty_price)
        assert result.empty

    def test_handles_none_price_features(self, sample_daily_sentiment):
        result = join_sentiment_with_prices(sample_daily_sentiment, None)
        assert result.empty

    def test_sorts_by_ticker_date(self, sample_price_features, sample_daily_sentiment):
        result = join_sentiment_with_prices(sample_daily_sentiment, sample_price_features)

        # Check sorting
        if not result.empty:
            dates_by_ticker = result.groupby("ticker")["date"].apply(lambda x: x.is_monotonic_increasing)
            assert dates_by_ticker.all()

    def test_preserves_topo_columns(self, sample_price_features, sample_daily_sentiment):
        # Ensure topo columns exist
        assert "topo_d0_total" in sample_price_features.columns

        result = join_sentiment_with_prices(sample_daily_sentiment, sample_price_features)

        assert "topo_d0_total" in result.columns
        assert "topo_d1_total" in result.columns
        assert "topo_entropy" in result.columns

    def test_fills_topo_na_with_zero(self, sample_price_features, sample_daily_sentiment):
        # Add some NaN to topo columns
        price_with_na = sample_price_features.copy()
        price_with_na.loc[0, "topo_d0_total"] = np.nan

        result = join_sentiment_with_prices(sample_daily_sentiment, price_with_na)

        # Should have filled NaN with 0
        assert not result["topo_d0_total"].isna().any()
