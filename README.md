# 📰📈 Sentiment Forecasting

[![CI](https://github.com/maniic/Sentiment-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/maniic/Sentiment-Forecasting/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)

**AI reads the day's financial headlines, topology measures the market's "shape", and a
machine-learning model turns both into trading signals — backtested against buy & hold
in an interactive web app.**

![App overview](docs/screenshots/app_overview.png)

## Try it in 30 seconds

```bash
git clone https://github.com/maniic/Sentiment-Forecasting.git
cd Sentiment-Forecasting
pip install -r requirements.txt
streamlit run app.py
```

The app opens in **Demo mode**: a fully offline, reproducible synthetic market where the
generated headlines genuinely predict next-day moves — so every chart has something to
show instantly, with no API keys, no waiting, and no model downloads. Flip one radio
button to **Live market data** to pull real headlines and real prices.

## How it works (in plain English)

| Step | What happens | Powered by |
|------|--------------|------------|
| 1. 📰 **Collect** | Pull the last few days of headlines for each ticker | Google News RSS / Yahoo Finance |
| 2. 🧠 **Read** | A finance-tuned AI reads each headline: positive / negative / neutral | FinBERT (lexicon fallback built in) |
| 3. 📐 **Shape** | Recent returns become a 3D point cloud; its *shape* is measured | Topological data analysis |
| 4. 🎲 **Predict** | Sentiment + momentum + shape → probability each ticker rises tomorrow | Logistic Regression / XGBoost / rule blend |
| 5. 📊 **Test** | Simulate trading that signal historically, minus costs, vs. buy & hold | Backtester |

### The sentiment view

Every headline is scored and aggregated per ticker per day — and you can inspect the
core hypothesis directly: does today's news tone predict tomorrow's move?

![Sentiment tab](docs/screenshots/app_sentiment.png)

### The topology view (the unusual part)

Most finance projects stop at returns and volatility. This one also applies
**persistent homology** — a technique from algebraic topology. A sliding window of
daily returns is *delay-embedded* into a point cloud, and the cloud's structure
(connected components H0, loops H1, and how long they persist) becomes a set of
`topo_*` features describing the market's regime: calm markets make tight simple
clouds, turbulent ones make stretched clouds with loops.

![Topology explorer](docs/screenshots/app_topology.png)

### Today's picks

The end product: tickers whose probability of rising tomorrow clears your threshold,
with the evidence (momentum, sentiment, headline count) beside each one.

![Today's picks](docs/screenshots/app_picks.png)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                │
├────────────────┬────────────────┬───────────────────────────────────┤
│  Google News   │   yfinance     │   Demo generator (offline,        │
│  RSS headlines │   news+prices  │   seeded synthetic market)        │
└───────┬────────┴───────┬────────┴───────────────┬───────────────────┘
        ▼                ▼                        ▼
┌─────────────────────────────────┐   ┌──────────────────────────────┐
│        SENTIMENT ENGINE         │   │     TOPOLOGICAL FEATURES     │
│  FinBERT transformer            │   │  Delay embedding             │
│  ↓ graceful fallback ↓          │   │  Vietoris-Rips persistence   │
│  Financial lexicon scorer       │   │  H0/H1 summaries + entropy   │
└───────────────┬─────────────────┘   └──────────────┬───────────────┘
                ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│   FEATURE ENGINEERING  (returns, volatility, sentiment, topology)   │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│   MODELS   rule blend │ Logistic Regression │ XGBoost │ Ensemble    │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│   BACKTEST   equal-weight longs · costs · drawdown · vs buy & hold  │
└─────────────────────────────────────────────────────────────────────┘
```

## Engineering highlights

- **Runs anywhere.** Heavy dependencies load lazily: no torch? The sentiment engine
  falls back to a transparent financial lexicon. No `giotto-tda`? Topology falls back
  to statistical proxies. No internet? Demo mode simulates the whole market. The app
  always reports which engine actually ran.
- **No look-ahead bias.** Signals formed on day *t* are evaluated on day *t+1* returns;
  train/test splits are strictly chronological.
- **Honest backtesting.** Trading costs, drawdown tracking, win rate, and an
  equal-weight buy-and-hold benchmark on every run.
- **Tested + CI.** 62 pytest tests covering schemas, features, news parsing, and models
  run on every push via GitHub Actions, plus an offline end-to-end pipeline smoke test.
- **Typed, modular `src/` layout** with centralized config, column-name contracts
  (`schemas.py`), and reusable UI components.

## Other ways to run it

```bash
# CLI pipeline — prints today's picks and saves equity curve + metrics to output/
python run_pipeline.py --tickers SPY QQQ AAPL
python run_pipeline.py --demo            # same, but fully offline

# Explore the notebook walkthrough
jupyter notebook notebooks/sentiment_forecasting_demo.ipynb

# Run the tests
pytest tests/ -v
```

## Configuration

All knobs live in `src/config.py`:

```python
from src.config import (
    TICKERS,               # Default ticker universe
    PREDICTION_THRESHOLD,  # Signal threshold (default: 0.60)
    DEFAULT_LOOKBACK_DAYS, # News lookback (default: 5)
    TopoConfig,            # TDA parameters
    ModelConfig,           # ML model settings
    StrategyConfig,        # Backtest parameters
)
```

## Project structure

```
├── app.py                  # Streamlit app (demo + live modes)
├── run_pipeline.py         # CLI pipeline
├── src/
│   ├── news.py             # Headline fetching (Google RSS, yfinance old+new schemas)
│   ├── sentiment.py        # FinBERT + lexicon fallback engines
│   ├── tda.py              # Persistent homology features (+ fallback)
│   ├── features.py         # Price/sentiment/topology feature engineering
│   ├── ml.py               # Models: rule blend, logistic, XGBoost, ensemble
│   ├── backtest.py         # Signals, backtester, benchmark, today's picks
│   ├── demo.py             # Seeded synthetic market generator
│   ├── config.py           # Central configuration
│   ├── schemas.py          # Column contracts shared across the pipeline
│   └── ui/                 # Streamlit components, Plotly charts, session state
├── tests/                  # 62 pytest tests
└── .github/workflows/      # CI: tests + offline pipeline smoke test
```

## Disclaimer

This is an educational project. Backtests on short windows are noisy, the demo market
is synthetic by design, and nothing here is investment advice.

## License

MIT — see [LICENSE](LICENSE).
