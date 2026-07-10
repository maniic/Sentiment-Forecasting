---
title: Sentiment Forecasting
emoji: 📰
short_description: FinBERT + topology → trading signals, backtested
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: true
license: mit
models:
  - yiyanghkust/finbert-tone
preload_from_hub:
  - yiyanghkust/finbert-tone
---

# 📰📈 Sentiment Forecasting

FinBERT reads the day's financial headlines, topological data analysis measures the
market's "shape", and a machine-learning model turns both into trading signals —
backtested against buy & hold in an interactive dashboard.

The Space serves the project's FastAPI dashboard directly. Source, tests, and docs:
**https://github.com/maniic/Sentiment-Forecasting**

> Educational demo — not investment advice.
