from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = ["ret_5", "ret_10", "vol_20", "sent_mean", "n"]

@dataclass
class MLReport:
    accuracy: float
    precision: float
    recall: float
    auc: float
    
def make_feature_table(df: pd.DataFrame, features: list[str] = None, label_col: str = "label") -> pd.DataFrame:
    """Return (x, y, dates), from the joined features DataFrame."""
    feats = features or DEFAULT_FEATURES
    keep = ["date"] + feats + [label_col]
    tbl = df[keep].dropna().copy()
    x = tbl[feats].astype(float)    
    y = tbl[label_col].astype(int)
    dates = pd.to_datetime(tbl["date"])
    return x, y, dates

def time_split_by_fraction(dates: pd.Series, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Time-based split: oldest (1-test_frac) for train, newest test_frac for test."""
    dates_sorted = np.sort(dates.unique())
    cut_idx = int(len(dates_sorted) * (1.0 -test_frac))
    cut_date = dates_sorted[max(0, cut_idx - 1)]
    train_mask = dates <= cut_date
    test_mask = dates > cut_date
    return train_mask.values, test_mask.values

def fit_logistic(X_train: pd.DataFrame, y_train: pd.Series, calibrate: bool = True, C: float = 1.0, class_weight: str | dict | None = None):
    """Standardized Logistic Regression"""
    base =  Pipeline([("scalar", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=2000, class_weight=class_weight))])
    if calibrate:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")),
        ])    
    else:
        model = base
    model.fit(X_train, y_train)
    return model

def predict_proba_up(model, X: pd.DataFrame) -> np.ndarray:
    """
    Return P(up) as the positive-class probability from a scikit-learn model.
    Works with LogisticRegression, CalibratedClassifierCV, or a Pipeline.
    """
    proba = model.predict_proba(X.astype(float))
    return proba[:, 1]

def add_ml_prob(df: pd.DataFrame, model, features: list[str] = None, out_col: str = "p_up_ml") -> pd.DataFrame:
    feats = features or DEFAULT_FEATURES
    out = df.copy()
    mask = out[feats].notna().all(axis=1)
    out.loc[mask, out_col] = predict_proba_up(model, out.loc[mask, feats].astype(float))
    # for rows missing features (rare), default to 0.5 so we don't bias
    out[out_col] = out[out_col].fillna(0.5)
    return out

def evaluate_probs(y_true: pd.Series, p_up: np.ndarray, threshold: float = 0.5) -> MLReport:
    y_pred = (p_up > threshold).astype(int)
    return MLReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        auc=float(roc_auc_score(y_true, p_up)),
    )
    
def save_model(model, path: str | Path ) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    
def load_model(path: str | Path):
    return joblib.load(path)
    

def add_quick_prob(
    df: pd.DataFrame,
    w_sent: float = 0.6,
    w_mom: float = 0.4,
    mom_scale: float = 10.0,
    temp: float = 3.0
) -> pd.DataFrame:
    """
    Adds a 'p_up' column using a simple blend:
        z = w_sent * sent_mean + w_mom * tanh(mom_scale * ret_5)
        p_up = sigmoid(temp * z)
    Returns a copy of df with p_up in [0, 1]
    """
    if df is None or df.empty:
        return df.assign(p_up=pd.Series(dtype="float64"))
    
    out = df.copy()
    
    # ensure numeric
    sent = out["sent_mean"].astype(float)
    mom = out["ret_5"].astype(float)
    
    z = w_sent * sent + w_mom * np.tanh(mom_scale * mom)
    out["p_up"] = 1.0 / (1.0 + np.exp(-temp * z))
    out["p_up"] = out["p_up"].clip(0.0, 1.0)
    return out

