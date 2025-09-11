from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import (
    TICKERS,
    DEFAULT_LOOKBACK_DAYS,
    PREDICTION_THRESHOLD,
    ROUND_TRIP_COST_BPS,
    NEWS_SOURCE,
)
from src.news import fetch_all_headlines
from src.sentiment import score_headlines, aggregate_daily_sentiment
from src.features import compute_price_features, join_sentiment_with_prices
from src.ml import add_quick_prob
from src.backtest import apply_signals, backtest_equal_weight, todays_signals
from src.visuals import plot_equity

def parse_args():
    p = argparse.ArgumentParser("News Sentiment Forecasting Pipeline")
    p.add_argument("--tickers", nargs="*", default=TICKERS)
    p.add_argument("--news-source", default=NEWS_SOURCE, choices=["google","yfinance","auto"])
    p.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    p.add_argument("--price-lookback", type=int, default=180)
    p.add_argument("--threshold", type=float, default=PREDICTION_THRESHOLD)
    p.add_argument("--require-mom", action="store_true", default=True)
    p.add_argument("--cost-bps", type=float, default=ROUND_TRIP_COST_BPS)
    p.add_argument("--outdir", default="output")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1) News → Sentiment
    news = fetch_all_headlines(args.tickers, args.lookback_days, source=args.news_source)
    scored = score_headlines(news)
    daily = aggregate_daily_sentiment(scored)

    # 2) Prices → Features
    price = compute_price_features(args.tickers, lookback_days=args.price_lookback)

    # 3) Join → quick probability
    X = join_sentiment_with_prices(daily, price)
    Xp = add_quick_prob(X)
    Xp.to_csv(outdir / "features_joined.csv", index=False)
    perf.to_csv(outdir / "equity_curve.csv", index=False)

    # 4) Signals + Backtest
    Xs = apply_signals(Xp, threshold=args.threshold, require_mom_agree=args.require_mom)
    perf, metrics = backtest_equal_weight(Xs, cost_bps=args.cost_bps)

    # 5) Today’s picks
    picks = todays_signals(Xp, threshold=args.threshold, require_mom_agree=args.require_mom)

    # 6) Save outputs
    picks_path = outdir / "predictions_today.csv"
    picks.to_csv(picks_path, index=False)

    metrics_path = outdir / "summary_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    eq_path = outdir / "equity_curve.png"
    plot_equity(perf, title="News+Momentum Threshold Strategy", save_path=str(eq_path))

    # 7) Console preview
    print("\n== Today’s Picks ==")
    if picks.empty:
        print("No signals today with the current threshold/filters.")
    else:
        print(picks.to_string(index=False))

    print("\n== Metrics ==")
    print(json.dumps(metrics, indent=2))

    print(f"\nSaved:\n  {picks_path}\n  {metrics_path}\n  {eq_path}")


if __name__ == "__main__":
    main()