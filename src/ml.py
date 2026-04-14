"""
Machine learning models for the Sentiment-Forecasting pipeline.

Provides model training, evaluation, and probability computation functions.
Supports logistic regression, XGBoost, and ensemble models.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, OUTPUT_DIR
from src.schemas import Columns

# Trusted directories that model artifacts may be loaded from.
# joblib.load deserializes pickle data, so any attacker-controlled
# path can lead to arbitrary code execution (CWE-502). We restrict
# load_model to a small, project-controlled allow-list to contain
# that risk.
_TRUSTED_MODEL_DIRS: tuple[Path, ...] = (
    Path(MODELS_DIR).resolve(),
    Path(OUTPUT_DIR).resolve(),
)

logger = logging.getLogger(__name__)

# Default feature set (auto-prunes missing cols)
DEFAULT_FEATURES = [
    Columns.RET_1,
    Columns.RET_5,
    Columns.RET_10,
    Columns.RET_20,
    Columns.VOL_20,
    Columns.SENT_MEAN,
    Columns.N_HEADLINES,
    Columns.TOPO_D0_TOTAL,
    Columns.TOPO_D1_TOTAL,
    Columns.TOPO_ENTROPY,
]

ModelType = Literal["quick", "logistic", "xgboost", "ensemble"]


@dataclass
class MLReport:
    """Report of ML model evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    auc: float


def select_available_features(df: pd.DataFrame, candidates: list[str] | None = None) -> list[str]:
    """Select features that exist in the DataFrame."""
    cands = candidates or DEFAULT_FEATURES
    return [c for c in cands if c in df.columns]


def make_feature_table(
    df: pd.DataFrame,
    features: list[str] | None = None,
    label_col: str = Columns.LABEL,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Prepare feature matrix and labels from joined features DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Joined features DataFrame
    features : list[str] | None
        Feature columns to use (defaults to DEFAULT_FEATURES)
    label_col : str
        Name of label column

    Returns
    -------
    tuple
        (X, y, dates, features_used)
    """
    feats = select_available_features(df, candidates=features or DEFAULT_FEATURES)
    keep = [Columns.DATE] + feats + [label_col]
    tbl = df[keep].dropna().copy()
    x = tbl[feats].astype(float)
    y = tbl[label_col].astype(int)
    dates = pd.to_datetime(tbl[Columns.DATE])
    return x, y, dates, feats


def time_split_by_fraction(
    dates: pd.Series,
    test_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Time-based train/test split.

    Parameters
    ----------
    dates : pd.Series
        Date series
    test_frac : float
        Fraction of data to use for testing

    Returns
    -------
    tuple
        (train_mask, test_mask) boolean arrays
    """
    dates_sorted = np.sort(dates.unique())
    cut_idx = int(len(dates_sorted) * (1.0 - test_frac))
    cut_date = dates_sorted[max(0, cut_idx - 1)]
    train_mask = dates <= cut_date
    test_mask = dates > cut_date
    return train_mask.values, test_mask.values


def fit_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    calibrate: bool = True,
    C: float = 1.0,
    class_weight: str | dict | None = None,
) -> Pipeline:
    """
    Train a standardized Logistic Regression model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    calibrate : bool
        Whether to use balanced class weights
    C : float
        Regularization parameter
    class_weight : str | dict | None
        Class weights for imbalanced data

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline
    """
    if calibrate:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")),
            ]
        )
    else:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=C, max_iter=2000, class_weight=class_weight)),
            ]
        )
    model.fit(X_train, y_train)
    return model


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
):
    """
    Train an XGBoost classifier.

    Falls back to HistGradientBoosting if XGBoost is unavailable.
    """
    try:
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=4,
            tree_method="hist",
        )
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logger.warning("XGBoost unavailable, falling back to HistGradientBoosting. err=%s", e)
        fallback = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05)
        fallback.fit(X_train, y_train)
        return fallback


def fit_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    voting: str = "soft",
) -> VotingClassifier:
    """
    Train an ensemble voting classifier combining logistic regression and gradient boosting.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    voting : str
        Voting strategy ('soft' or 'hard')

    Returns
    -------
    VotingClassifier
        Fitted ensemble model
    """
    # Logistic regression with scaling
    lr_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")),
        ]
    )

    # Gradient boosting
    gb_model = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=200,
        class_weight="balanced",
    )

    # Try to add XGBoost if available
    estimators = [("lr", lr_model), ("gb", gb_model)]

    try:
        import xgboost as xgb

        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=4,
            tree_method="hist",
        )
        estimators.append(("xgb", xgb_model))
        logger.info("Ensemble includes: LogisticRegression, HistGradientBoosting, XGBoost")
    except ImportError:
        logger.info("Ensemble includes: LogisticRegression, HistGradientBoosting (XGBoost not available)")

    ensemble = VotingClassifier(estimators=estimators, voting=voting)
    ensemble.fit(X_train, y_train)
    return ensemble


def _predict_proba_any(model, X: pd.DataFrame) -> np.ndarray:
    """Get probability predictions from any model type."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X.astype(float))
        return proba[:, -1]
    if hasattr(model, "predict"):
        pred = model.predict(X.astype(float))
        if pred.ndim == 1:
            return pred
        return pred[:, -1]
    raise ValueError("Model does not support predict_proba or predict for probability output")


def add_ml_prob(
    df: pd.DataFrame,
    model,
    features: list[str] | None = None,
    out_col: str = Columns.P_UP_ML,
) -> pd.DataFrame:
    """
    Add ML probability predictions to DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    model : object
        Fitted model with predict_proba method
    features : list[str] | None
        Feature columns to use
    out_col : str
        Name of output probability column

    Returns
    -------
    pd.DataFrame
        DataFrame with probability column added
    """
    feats = select_available_features(df, features)
    out = df.copy()
    if not feats:
        out[out_col] = 0.5
        return out
    mask = out[feats].notna().all(axis=1)
    out.loc[mask, out_col] = _predict_proba_any(model, out.loc[mask, feats].astype(float))
    out[out_col] = out[out_col].fillna(0.5)
    return out


def evaluate_probs(
    y_true: pd.Series,
    p_up: np.ndarray,
    threshold: float = 0.5,
) -> MLReport:
    """Evaluate probability predictions against true labels."""
    y_pred = (p_up > threshold).astype(int)
    return MLReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        auc=float(roc_auc_score(y_true, p_up)),
    )


def _best_threshold(
    y_true: pd.Series,
    p_up: np.ndarray,
    grid: Sequence[float] | None = None,
) -> float:
    """Find the threshold that maximizes F1 score."""
    grid = grid or np.linspace(0.45, 0.65, 11)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (p_up > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t)


def _resolve_trusted_model_path(
    path: str | Path,
    allowed_dirs: tuple[Path, ...] = _TRUSTED_MODEL_DIRS,
) -> Path:
    """
    Resolve ``path`` and ensure it lives under one of ``allowed_dirs``.

    Raises
    ------
    ValueError
        If the resolved path escapes every allowed directory.
    """
    resolved = Path(path).expanduser().resolve()
    for root in allowed_dirs:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    allowed_str = ", ".join(str(d) for d in allowed_dirs)
    raise ValueError(
        f"Refusing to load model from untrusted path: {resolved!s}. "
        f"Allowed roots: {allowed_str}"
    )


def save_model(model, path: str | Path) -> None:
    """Save model to disk using joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    """
    Load a model from disk using joblib.

    ``joblib.load`` deserializes pickle data, which can execute arbitrary
    code. To mitigate CWE-502 (Deserialization of Untrusted Data), the
    provided path must resolve inside one of the project's trusted
    directories (see ``_TRUSTED_MODEL_DIRS``). Paths that escape via
    symlinks, ``..`` traversal, or absolute locations are rejected.
    """
    safe_path = _resolve_trusted_model_path(path)
    return joblib.load(safe_path)


def train_model_from_frame(
    df: pd.DataFrame,
    model_type: ModelType = "logistic",
    features: list[str] | None = None,
    test_frac: float = 0.2,
    calibrate: bool = False,
) -> tuple[object, MLReport, float, list[str]]:
    """
    Train a model on the joined feature frame.

    Parameters
    ----------
    df : pd.DataFrame
        Joined features DataFrame
    model_type : ModelType
        Type of model to train
    features : list[str] | None
        Feature columns to use
    test_frac : float
        Fraction of data for testing
    calibrate : bool
        Whether to use balanced class weights

    Returns
    -------
    tuple
        (model, test_report, threshold, features_used)
    """
    X, y, dates, feats = make_feature_table(df, features=features)
    if X.empty:
        raise ValueError("No data available to train model")

    train_mask, test_mask = time_split_by_fraction(dates, test_frac=test_frac)
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]

    logger.info("Training %s model: train=%d, test=%d, features=%s", model_type, len(Xtr), len(Xte), feats)

    if model_type == "logistic":
        model = fit_logistic(Xtr, ytr, calibrate=calibrate, C=0.1, class_weight="balanced")
    elif model_type == "xgboost":
        model = fit_xgboost(Xtr, ytr)
    elif model_type == "ensemble":
        model = fit_ensemble(Xtr, ytr)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    p_tr = _predict_proba_any(model, Xtr)
    threshold = _best_threshold(ytr, p_tr)
    p_te = _predict_proba_any(model, Xte)
    report = evaluate_probs(yte, p_te, threshold=threshold)

    logger.info("Model trained. Test AUC=%.3f, threshold=%.2f", report.auc, threshold)

    return model, report, threshold, feats


def add_quick_prob(
    df: pd.DataFrame,
    w_sent: float = 0.6,
    w_mom: float = 0.4,
    mom_scale: float = 10.0,
    temp: float = 3.0,
) -> pd.DataFrame:
    """
    Add a 'p_up' column using a simple blend of sentiment and momentum.

    Formula:
        z = w_sent * sent_mean + w_mom * tanh(mom_scale * ret_5)
        p_up = sigmoid(temp * z)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with sent_mean and ret_5 columns
    w_sent : float
        Weight for sentiment
    w_mom : float
        Weight for momentum
    mom_scale : float
        Scaling factor for momentum
    temp : float
        Temperature for sigmoid

    Returns
    -------
    pd.DataFrame
        DataFrame with p_up column in [0, 1]
    """
    if df is None:
        return None
    if df.empty:
        return df.assign(**{Columns.P_UP: pd.Series(dtype="float64")})

    out = df.copy()

    sent = out[Columns.SENT_MEAN].astype(float)
    mom = out[Columns.RET_5].astype(float)

    z = w_sent * sent + w_mom * np.tanh(mom_scale * mom)
    out[Columns.P_UP] = 1.0 / (1.0 + np.exp(-temp * z))
    out[Columns.P_UP] = out[Columns.P_UP].clip(0.0, 1.0)
    return out


def blend_probabilities(
    probs: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Blend multiple probability arrays with optional weights."""
    if not probs:
        return np.array([])
    w = np.array(weights if weights is not None else [1.0 / len(probs)] * len(probs), dtype=float)
    w = w / w.sum()
    stacked = np.vstack(probs)
    return np.dot(w, stacked)
