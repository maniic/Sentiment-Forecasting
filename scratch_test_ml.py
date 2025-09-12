# scratch_test_ml.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
)
from sklearn.calibration import CalibrationDisplay
import joblib

from src.config import (
    TICKERS, DEFAULT_LOOKBACK_DAYS, PREDICTION_THRESHOLD, ROUND_TRIP_COST_BPS
)
from src.news import fetch_all_headlines
from src.sentiment import score_headlines, aggregate_daily_sentiment
from src.features import compute_price_features, join_sentiment_with_prices
from src.ml import add_quick_prob
from src.backtest import apply_signals, backtest_equal_weight


def choose_available_features(df: pd.DataFrame) -> list[str]:
    # Will use what exists in your current join schema
    candidates = ["ret_1", "ret_5", "ret_10", "vol_20", "sent_mean", "sent_abs", "n"]
    feats = [c for c in candidates if c in df.columns]
    if len(feats) < 3:
        raise ValueError(f"Not enough features present. Found: {feats}")
    return feats


def time_split_by_fraction(dates: pd.Series, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.to_datetime(dates)
    uniq = np.sort(dates.unique())
    cut_idx = int(len(uniq) * (1.0 - test_frac))
    cut_date = uniq[max(0, cut_idx - 1)]
    train_mask = dates <= cut_date
    test_mask = dates > cut_date
    return train_mask.values, test_mask.values


def main():
    outdir = Path("output"); outdir.mkdir(parents=True, exist_ok=True)
    tickers = TICKERS[:5]

    # 1) Build dataset
    print(">> Building dataset...")
    news = fetch_all_headlines(tickers, DEFAULT_LOOKBACK_DAYS)
    scored = score_headlines(news)
    daily = aggregate_daily_sentiment(scored)
    price = compute_price_features(tickers, lookback_days=180)
    X_join = join_sentiment_with_prices(daily, price)

    # 2) Feature table (robust to available columns)
    feats = choose_available_features(X_join)
    keep_cols = ["date"] + feats + ["label"]
    tbl = X_join[keep_cols].dropna().copy()
    X = tbl[feats].astype(float)
    y = tbl["label"].astype(int)
    dates = pd.to_datetime(tbl["date"])

    train_mask, test_mask = time_split_by_fraction(dates, test_frac=0.2)
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    print(f">> Train: {Xtr.shape}, Test: {Xte.shape}, Features: {feats}")

    # 3) Train logistic baseline (balanced, mild regularization)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")),
    ])
    model.fit(Xtr, ytr)

    # 4) Pick threshold on TRAIN (F1 grid)
    p_tr = model.predict_proba(Xtr)[:, 1]
    cands = np.linspace(0.45, 0.65, 9)
    best_t, best_f1 = 0.50, -1.0
    for t in cands:
        f1 = f1_score(ytr, (p_tr > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    print(f">> Chosen train threshold t*={best_t:.2f} (F1={best_f1:.3f})")

    # 5) Evaluate on TEST
    p_te = model.predict_proba(Xte)[:, 1]
    acc = accuracy_score(yte, (p_te > best_t).astype(int))
    prec = precision_score(yte, (p_te > best_t).astype(int), zero_division=0)
    rec = recall_score(yte, (p_te > best_t).astype(int), zero_division=0)
    try:
        auc = roc_auc_score(yte, p_te)
    except ValueError:
        auc = float("nan")
    print(">> Test metrics:")
    print(json.dumps({"accuracy": acc, "precision": prec, "recall": rec, "auc": auc}, indent=2))

    # 6) Calibration curve
    fig = plt.figure()
    CalibrationDisplay.from_predictions(yte, p_te, n_bins=10)
    plt.title("Calibration (Logistic)")
    plt.tight_layout()
    cal_path = outdir / "calibration_curve.png"
    plt.savefig(cal_path, dpi=150)
    plt.close(fig)
    print(">> Saved:", cal_path)

    # 7) Backtests — ML vs Rule-based
    # ML P(up) over the full joined frame
    X_ml = X_join.copy()
    mask = X_ml[feats].notna().all(axis=1)
    X_ml.loc[mask, "p_up_ml"] = model.predict_proba(X_ml.loc[mask, feats].astype(float))[:, 1]
    X_ml["p_up_ml"] = X_ml["p_up_ml"].fillna(0.5)

    Xs_ml = apply_signals(X_ml, p_col="p_up_ml", threshold=best_t, require_mom_agree=True)
    perf_ml, metrics_ml = backtest_equal_weight(Xs_ml, cost_bps=ROUND_TRIP_COST_BPS)

    # Rule-based (existing quick blend)
    X_rb = add_quick_prob(X_join)
    Xs_rb = apply_signals(X_rb, p_col="p_up", threshold=PREDICTION_THRESHOLD, require_mom_agree=True)
    perf_rb, metrics_rb = backtest_equal_weight(Xs_rb, cost_bps=ROUND_TRIP_COST_BPS)

    print("\n>> Compare Backtests (net):")
    print("  ML  :", json.dumps(metrics_ml, indent=2))
    print("  Rule:", json.dumps(metrics_rb, indent=2))

    # 8) Save model
    model_path = outdir / "logreg_sentiment.joblib"
    joblib.dump(model, model_path)
    print(">> Saved model to:", model_path)

    print("\nML TEST PASSED ✅")


if __name__ == "__main__":
    main()
