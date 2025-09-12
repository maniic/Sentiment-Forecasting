# Sentiment-Forecasting
A sentiment-driven stock forecasting tool that combines FinBERT headline analysis with price momentum and logistic regression.

**Ask about a stock.** We pull recent headlines, score them with **FinBERT**, blend with **price momentum**, and generate **next-day up probabilities**. Then we **threshold**, **backtest**, and show **today’s picks**.

<div align="left">

- FinBERT sentiment (Hugging Face)  
- Price features (ret_5, ret_10, vol_20, …)  
- Quick probability blend **or** trained Logistic Regression  
- Threshold strategy with costs, equity curve, Sharpe & drawdown  
- CLI, Notebook, and GUI

</div>

---