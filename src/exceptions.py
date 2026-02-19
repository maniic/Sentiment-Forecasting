"""
Custom exceptions for the Sentiment-Forecasting pipeline.

Provides a hierarchy of exceptions for better error handling and debugging.
"""
from __future__ import annotations


class SentimentForecastingError(Exception):
    """Base exception for all Sentiment-Forecasting errors."""

    pass


class DataFetchError(SentimentForecastingError):
    """Raised when data fetching fails (news, prices, etc.)."""

    pass


class ValidationError(SentimentForecastingError):
    """Raised when data validation fails."""

    pass


class ModelError(SentimentForecastingError):
    """Raised when model training or prediction fails."""

    pass


class ConfigurationError(SentimentForecastingError):
    """Raised when configuration is invalid."""

    pass


class InsufficientDataError(SentimentForecastingError):
    """Raised when there is not enough data to perform an operation."""

    pass
