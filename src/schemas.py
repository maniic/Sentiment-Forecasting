"""
Data contracts and column name constants for the Sentiment-Forecasting pipeline.

This module centralizes all column names and provides validation utilities
to ensure consistency across the pipeline stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd


class Columns:
    """Centralized column name constants used across the pipeline."""

    # Identifiers
    TICKER = "ticker"
    DATE = "date"

    # News/Headlines
    TIMESTAMP = "ts"
    HEADLINE = "headline"
    SOURCE = "source"
    LINK = "link"

    # Sentiment
    SENT_LABEL = "sent_label"
    SENT_PROB = "sent_prob"
    SENT_SCORE = "sent_score"
    SENT_MEAN = "sent_mean"
    SENT_STD = "sent_std"
    N_HEADLINES = "n"

    # Price features
    RET_1 = "ret_1"
    RET_5 = "ret_5"
    RET_10 = "ret_10"
    RET_20 = "ret_20"
    VOL_20 = "vol_20"
    NEXT_RET = "next_ret"
    LABEL = "label"

    # Topology features
    TOPO_D0_TOTAL = "topo_d0_total"
    TOPO_D0_MAX = "topo_d0_max"
    TOPO_D0_MEAN = "topo_d0_mean"
    TOPO_D0_COUNT = "topo_d0_count"
    TOPO_D1_TOTAL = "topo_d1_total"
    TOPO_D1_MAX = "topo_d1_max"
    TOPO_D1_MEAN = "topo_d1_mean"
    TOPO_D1_COUNT = "topo_d1_count"
    TOPO_ENTROPY = "topo_entropy"

    # Probability/Signal
    P_UP = "p_up"
    P_UP_ML = "p_up_ml"
    LONG = "long"

    # Backtest
    N_POSITIONS = "n_positions"
    GROSS = "gross"
    NET = "net"
    EQUITY = "equity"


# Column groups for validation
NEWS_COLUMNS = [Columns.TICKER, Columns.TIMESTAMP, Columns.HEADLINE, Columns.SOURCE, Columns.LINK]

SENTIMENT_COLUMNS = [Columns.SENT_LABEL, Columns.SENT_PROB, Columns.SENT_SCORE]

DAILY_SENTIMENT_COLUMNS = [Columns.TICKER, Columns.DATE, Columns.SENT_MEAN, Columns.SENT_STD, Columns.N_HEADLINES]

PRICE_FEATURE_COLUMNS = [
    Columns.TICKER,
    Columns.DATE,
    Columns.RET_1,
    Columns.RET_5,
    Columns.RET_10,
    Columns.RET_20,
    Columns.VOL_20,
    Columns.NEXT_RET,
    Columns.LABEL,
]

TOPO_FEATURE_COLUMNS = [
    Columns.TOPO_D0_TOTAL,
    Columns.TOPO_D0_MAX,
    Columns.TOPO_D0_MEAN,
    Columns.TOPO_D0_COUNT,
    Columns.TOPO_D1_TOTAL,
    Columns.TOPO_D1_MAX,
    Columns.TOPO_D1_MEAN,
    Columns.TOPO_D1_COUNT,
    Columns.TOPO_ENTROPY,
]


@dataclass(frozen=True)
class NewsSchema:
    """Schema for raw news/headline data."""

    required_columns: ClassVar[list[str]] = NEWS_COLUMNS

    @classmethod
    def validate(cls, df: pd.DataFrame) -> bool:
        """Check if DataFrame has all required columns."""
        if df is None or df.empty:
            return True  # Empty is valid
        return all(col in df.columns for col in cls.required_columns)

    @classmethod
    def empty(cls) -> pd.DataFrame:
        """Return an empty DataFrame with correct schema."""
        return pd.DataFrame(columns=cls.required_columns)


@dataclass(frozen=True)
class SentimentSchema:
    """Schema for scored sentiment data (news + sentiment scores)."""

    required_columns: ClassVar[list[str]] = NEWS_COLUMNS + SENTIMENT_COLUMNS

    @classmethod
    def validate(cls, df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return True
        return all(col in df.columns for col in cls.required_columns)


@dataclass(frozen=True)
class DailySentimentSchema:
    """Schema for aggregated daily sentiment."""

    required_columns: ClassVar[list[str]] = DAILY_SENTIMENT_COLUMNS

    @classmethod
    def validate(cls, df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return True
        return all(col in df.columns for col in cls.required_columns)

    @classmethod
    def empty(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls.required_columns)


@dataclass(frozen=True)
class PriceFeatureSchema:
    """Schema for price feature data."""

    required_columns: ClassVar[list[str]] = PRICE_FEATURE_COLUMNS

    @classmethod
    def validate(cls, df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return True
        return all(col in df.columns for col in cls.required_columns)

    @classmethod
    def empty(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls.required_columns)


@dataclass(frozen=True)
class JoinedFeatureSchema:
    """Schema for joined sentiment + price features."""

    base_columns: ClassVar[list[str]] = [
        Columns.TICKER,
        Columns.DATE,
        Columns.RET_1,
        Columns.RET_5,
        Columns.RET_10,
        Columns.RET_20,
        Columns.VOL_20,
        Columns.SENT_MEAN,
        Columns.SENT_STD,
        Columns.N_HEADLINES,
        Columns.NEXT_RET,
        Columns.LABEL,
    ]

    @classmethod
    def validate(cls, df: pd.DataFrame, require_topo: bool = False) -> bool:
        if df is None or df.empty:
            return True
        base_valid = all(col in df.columns for col in cls.base_columns)
        if require_topo:
            return base_valid and all(col in df.columns for col in TOPO_FEATURE_COLUMNS)
        return base_valid
