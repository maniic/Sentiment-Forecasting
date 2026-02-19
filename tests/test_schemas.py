"""Tests for schemas module."""
from __future__ import annotations

import pandas as pd
import pytest

from src.schemas import (
    Columns,
    NewsSchema,
    DailySentimentSchema,
    PriceFeatureSchema,
    NEWS_COLUMNS,
    DAILY_SENTIMENT_COLUMNS,
)


class TestColumns:
    """Tests for column name constants."""

    def test_ticker_constant(self):
        assert Columns.TICKER == "ticker"

    def test_date_constant(self):
        assert Columns.DATE == "date"

    def test_sentiment_constants(self):
        assert Columns.SENT_MEAN == "sent_mean"
        assert Columns.SENT_STD == "sent_std"

    def test_price_feature_constants(self):
        assert Columns.RET_1 == "ret_1"
        assert Columns.RET_5 == "ret_5"
        assert Columns.VOL_20 == "vol_20"

    def test_topo_constants(self):
        assert Columns.TOPO_D0_TOTAL == "topo_d0_total"
        assert Columns.TOPO_ENTROPY == "topo_entropy"


class TestNewsSchema:
    """Tests for news schema validation."""

    def test_validate_valid_df(self, sample_news_df):
        assert NewsSchema.validate(sample_news_df)

    def test_validate_empty_df(self):
        assert NewsSchema.validate(pd.DataFrame())

    def test_validate_none(self):
        assert NewsSchema.validate(None)

    def test_validate_missing_columns(self):
        df = pd.DataFrame({"ticker": ["AAPL"]})  # Missing other columns
        assert not NewsSchema.validate(df)

    def test_empty_creates_correct_schema(self):
        df = NewsSchema.empty()
        assert list(df.columns) == NEWS_COLUMNS
        assert len(df) == 0


class TestDailySentimentSchema:
    """Tests for daily sentiment schema validation."""

    def test_validate_valid_df(self, sample_daily_sentiment):
        assert DailySentimentSchema.validate(sample_daily_sentiment)

    def test_validate_empty_df(self):
        assert DailySentimentSchema.validate(pd.DataFrame())

    def test_empty_creates_correct_schema(self):
        df = DailySentimentSchema.empty()
        assert list(df.columns) == DAILY_SENTIMENT_COLUMNS


class TestPriceFeatureSchema:
    """Tests for price feature schema validation."""

    def test_validate_valid_df(self, sample_price_features):
        assert PriceFeatureSchema.validate(sample_price_features)

    def test_validate_empty_df(self):
        assert PriceFeatureSchema.validate(pd.DataFrame())

    def test_empty_creates_correct_schema(self):
        df = PriceFeatureSchema.empty()
        expected_cols = [
            "ticker", "date", "ret_1", "ret_5", "ret_10",
            "ret_20", "vol_20", "next_ret", "label"
        ]
        assert list(df.columns) == expected_cols
