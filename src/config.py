TICKERS = [
    "SPY",
    "QQQ",
    "AAPL",
    "TSLA", 
    "AMD",
    "AMZN",
    "NVDA",
    "META",
    "MSFT",
    "GOOGL"
    ]

# News lookback
LOOKBACK_WINDOWS = [3, 5, 7] # benchmark values
DEFAULT_LOOKBACK_DAYS = 5    # default value

# Strategy
PREDICTION_THRESHOLD = 0.60
ROUND_TRIP_COST_BPS = 3

NEWS_SOURCE = "auto" # 'google' or 'yfinance' or 'auto' (try google, fallback to yfinance)
