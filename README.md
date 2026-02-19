# Sentiment Forecasting

A sophisticated stock sentiment forecasting system that combines **FinBERT NLP**, **Topological Data Analysis (TDA)**, and **ML-driven backtesting** to generate trading signals.

## Overview

This is a stock prediction project that combines three cutting-edge techniques:

### 1. FinBERT Financial Sentiment Analysis
- State-of-the-art transformer model fine-tuned on financial text
- Classifies headlines as POSITIVE, NEGATIVE, or NEUTRAL with calibrated probabilities
- Outperforms generic sentiment models on financial language

### 2. Topological Data Analysis (TDA)
- Uses **persistent homology** from algebraic topology to detect hidden market patterns
- Transforms price series into point clouds via **delay embedding**
- Extracts topological features (connected components, loops) that capture market regimes
- This is a rare, advanced technique seldom seen in finance projects

### 3. End-to-End ML Pipeline
- Time-series aware train/test splitting (no look-ahead bias)
- Multiple model support: Logistic Regression, XGBoost, Ensemble
- Full backtesting with transaction costs
- Real-time signal generation

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                │
├──────────────────┬──────────────────┬───────────────────────────────┤
│   Google News    │    yfinance      │       yfinance OHLCV          │
│   Headlines      │    Headlines     │       Price Data              │
└────────┬─────────┴────────┬─────────┴─────────────┬─────────────────┘
         │                  │                       │
         ▼                  ▼                       ▼
┌─────────────────────────────────┐    ┌──────────────────────────────┐
│         FinBERT NLP             │    │    Topological Features      │
│  ┌───────────────────────────┐  │    │  ┌────────────────────────┐  │
│  │ Sentiment Classification  │  │    │  │  Delay Embedding       │  │
│  │ POSITIVE / NEGATIVE /     │  │    │  │  Vietoris-Rips Complex │  │
│  │ NEUTRAL with confidence   │  │    │  │  Persistence Diagrams  │  │
│  └───────────────────────────┘  │    │  └────────────────────────┘  │
└────────────────┬────────────────┘    └─────────────┬────────────────┘
                 │                                   │
                 ▼                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                            │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌────────────┐  │
│  │ sent_mean   │  │ ret_1/5/10   │  │ topo_d0_*  │  │ topo_d1_*  │  │
│  │ sent_std    │  │ vol_20       │  │ topo_entropy│ │            │  │
│  └─────────────┘  └──────────────┘  └────────────┘  └────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ML MODELS                                    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│  │  Logistic   │  │   XGBoost    │  │   Ensemble (Voting)        │  │
│  │  Regression │  │              │  │   LR + GB + XGB            │  │
│  └─────────────┘  └──────────────┘  └────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BACKTESTING                                    │
│  • Time-series cross-validation    • Transaction cost modeling     │
│  • Sharpe ratio optimization       • Maximum drawdown tracking     │
│  • Signal threshold tuning         • Today's live signals          │
└─────────────────────────────────────────────────────────────────────┘
```

### Run the Pipeline

```bash
# CLI Pipeline - generates signals and backtests
python run_pipeline.py --tickers SPY QQQ AAPL

# Launch the Streamlit GUI
streamlit run app.py
```

### Explore the Demo Notebook

```bash
# Launch Jupyter
jupyter notebook notebooks/sentiment_forecasting_demo.ipynb
```

## Features

### Data Sources
- **Google News RSS**: Real-time financial headlines
- **yfinance**: Stock prices and alternative news source
- **Automatic fallback**: Switches sources if one fails

### Sentiment Analysis
- **FinBERT**: Domain-specific financial sentiment model
- **Thread-safe singleton**: Efficient model loading
- **Batch processing**: Memory-efficient inference

### Topological Features
- **Delay embedding**: Time series → point cloud transformation
- **Persistent homology**: H0 (components) and H1 (loops) features
- **Fallback statistics**: Works even without giotto-tda installed

### ML Models
- **Quick Prob**: Rule-based sentiment + momentum blend
- **Logistic Regression**: Balanced, regularized classifier
- **XGBoost**: Gradient boosting with auto-fallback
- **Ensemble**: Voting classifier combining all models

### Backtesting
- **Equal-weight strategy**: Long signals above threshold
- **Momentum filter**: Optional trend confirmation
- **Transaction costs**: Realistic cost modeling
- **Metrics**: Sharpe ratio, max drawdown, trade count

## Configuration

All configuration is centralized in `src/config.py`:

```python
from src.config import (
    TICKERS,               # Default ticker universe
    PREDICTION_THRESHOLD,  # Signal threshold (default: 0.60)
    DEFAULT_LOOKBACK_DAYS, # News lookback (default: 5)
    TopoConfig,           # TDA parameters
    ModelConfig,          # ML model settings
    StrategyConfig,       # Backtest parameters
)
```

## Dependencies

Core:
- `pandas`, `numpy` - Data manipulation
- `transformers`, `torch` - FinBERT model
- `scikit-learn`, `xgboost` - ML models
- `giotto-tda` - Topological Data Analysis
- `yfinance`, `feedparser` - Data fetching
- `streamlit` - Web interface
- `matplotlib`, `plotly` - Visualization
