"""Tests for ML module."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from src.config import MODELS_DIR, OUTPUT_DIR
from src.ml import (
    select_available_features,
    add_quick_prob,
    blend_probabilities,
    time_split_by_fraction,
    load_model,
    _resolve_trusted_model_path,
    MLReport,
    DEFAULT_FEATURES,
)
from src.schemas import Columns


class TestSelectAvailableFeatures:
    """Tests for feature selection."""

    def test_selects_existing_columns(self):
        df = pd.DataFrame({
            "ret_1": [1, 2],
            "ret_5": [3, 4],
            "other": [5, 6],
        })
        result = select_available_features(df, candidates=["ret_1", "ret_5", "missing"])
        assert result == ["ret_1", "ret_5"]

    def test_returns_empty_for_no_matches(self):
        df = pd.DataFrame({"other": [1, 2]})
        result = select_available_features(df, candidates=["ret_1", "ret_5"])
        assert result == []

    def test_uses_default_features_when_none(self):
        df = pd.DataFrame({col: [1, 2] for col in DEFAULT_FEATURES[:3]})
        result = select_available_features(df, candidates=None)
        assert len(result) == 3


class TestAddQuickProb:
    """Tests for quick probability computation."""

    def test_adds_p_up_column(self):
        df = pd.DataFrame({
            "sent_mean": [0.5, -0.5, 0.0],
            "ret_5": [0.01, -0.01, 0.0],
        })
        result = add_quick_prob(df)
        assert Columns.P_UP in result.columns

    def test_p_up_in_range(self):
        df = pd.DataFrame({
            "sent_mean": [1.0, -1.0, 0.0, 0.5],
            "ret_5": [0.1, -0.1, 0.0, 0.05],
        })
        result = add_quick_prob(df)
        assert (result[Columns.P_UP] >= 0.0).all()
        assert (result[Columns.P_UP] <= 1.0).all()

    def test_positive_sentiment_higher_prob(self):
        df = pd.DataFrame({
            "sent_mean": [0.8, -0.8],
            "ret_5": [0.0, 0.0],  # No momentum effect
        })
        result = add_quick_prob(df)
        assert result[Columns.P_UP].iloc[0] > result[Columns.P_UP].iloc[1]

    def test_positive_momentum_higher_prob(self):
        df = pd.DataFrame({
            "sent_mean": [0.0, 0.0],  # No sentiment effect
            "ret_5": [0.05, -0.05],
        })
        result = add_quick_prob(df)
        assert result[Columns.P_UP].iloc[0] > result[Columns.P_UP].iloc[1]

    def test_handles_empty_df(self):
        df = pd.DataFrame(columns=["sent_mean", "ret_5"])
        result = add_quick_prob(df)
        assert Columns.P_UP in result.columns
        assert len(result) == 0

    def test_handles_none(self):
        result = add_quick_prob(None)
        assert result is None


class TestBlendProbabilities:
    """Tests for probability blending."""

    def test_equal_weights(self):
        probs = [np.array([0.6, 0.4]), np.array([0.8, 0.2])]
        result = blend_probabilities(probs)
        expected = np.array([0.7, 0.3])
        np.testing.assert_array_almost_equal(result, expected)

    def test_custom_weights(self):
        probs = [np.array([0.6, 0.4]), np.array([0.8, 0.2])]
        result = blend_probabilities(probs, weights=[0.75, 0.25])
        # 0.75 * 0.6 + 0.25 * 0.8 = 0.45 + 0.2 = 0.65
        assert result[0] == pytest.approx(0.65)

    def test_empty_list(self):
        result = blend_probabilities([])
        assert len(result) == 0


class TestTimeSplitByFraction:
    """Tests for time-based train/test split."""

    def test_splits_by_date(self):
        dates = pd.Series(pd.date_range("2024-01-01", periods=100))
        train_mask, test_mask = time_split_by_fraction(dates, test_frac=0.2)

        # Should have ~80 train, ~20 test
        assert train_mask.sum() >= 75
        assert test_mask.sum() >= 15
        assert train_mask.sum() + test_mask.sum() == 100

    def test_no_overlap(self):
        dates = pd.Series(pd.date_range("2024-01-01", periods=100))
        train_mask, test_mask = time_split_by_fraction(dates, test_frac=0.2)

        # No overlap
        assert not (train_mask & test_mask).any()

    def test_train_before_test(self):
        dates = pd.Series(pd.date_range("2024-01-01", periods=100))
        train_mask, test_mask = time_split_by_fraction(dates, test_frac=0.2)

        # All train dates should be before all test dates
        train_max = dates[train_mask].max()
        test_min = dates[test_mask].min()
        assert train_max < test_min


class TestMLReport:
    """Tests for ML report dataclass."""

    def test_creation(self):
        report = MLReport(accuracy=0.8, precision=0.7, recall=0.6, auc=0.85)
        assert report.accuracy == 0.8
        assert report.precision == 0.7
        assert report.recall == 0.6
        assert report.auc == 0.85


class TestLoadModelTrustedPath:
    """Guard against CWE-502 regressions in load_model."""

    def test_allows_path_inside_models_dir(self, tmp_path, monkeypatch):
        rel = "model_lr.joblib"
        resolved = _resolve_trusted_model_path(Path(MODELS_DIR) / rel)
        assert resolved == (Path(MODELS_DIR).resolve() / rel)

    def test_allows_path_inside_output_dir(self):
        rel = "logreg_sentiment.joblib"
        resolved = _resolve_trusted_model_path(Path(OUTPUT_DIR) / rel)
        assert resolved == (Path(OUTPUT_DIR).resolve() / rel)

    def test_rejects_absolute_outside_trusted_dirs(self):
        with pytest.raises(ValueError, match="untrusted path"):
            _resolve_trusted_model_path("/etc/passwd")

    def test_rejects_parent_traversal(self):
        with pytest.raises(ValueError, match="untrusted path"):
            _resolve_trusted_model_path("output/../../../tmp/evil.joblib")

    def test_load_model_refuses_untrusted_path(self, tmp_path):
        # Create a file outside the allowed directories and ensure
        # load_model refuses rather than deserializing it.
        evil = tmp_path / "evil.joblib"
        evil.write_bytes(b"not really a pickle")
        with pytest.raises(ValueError, match="untrusted path"):
            load_model(evil)
