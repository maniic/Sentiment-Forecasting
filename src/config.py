"""
Centralized configuration for the Sentiment-Forecasting pipeline.

All configuration constants and dataclasses are defined here to provide
a single source of truth for pipeline settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# =============================================================================
# Tickers
# =============================================================================
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
    "GOOGL",
]

# =============================================================================
# News Configuration
# =============================================================================
LOOKBACK_WINDOWS = [3, 5, 7]  # benchmark values
DEFAULT_LOOKBACK_DAYS = 5
MAX_HEADLINES_PER_TICKER = 50
NEWS_SOURCE: Literal["google", "yfinance", "auto"] = "auto"

# =============================================================================
# Strategy Configuration
# =============================================================================
PREDICTION_THRESHOLD = 0.60
ROUND_TRIP_COST_BPS = 3

# Backtesting
SLIPPAGE_BPS = 1.0
SPREAD_BPS = 2.0
MAX_GROSS_EXPOSURE = 1.0  # fully invested cap
VOL_TARGET = 0.10  # annualized vol target for scaling

# =============================================================================
# Topology (TDA) Configuration
# =============================================================================
ENABLE_TOPOLOGY = True
TOPO_WINDOW = 60
TOPO_EMBED_DIM = 4


@dataclass(frozen=True)
class TopoConfig:
    """Configuration for Topological Data Analysis features."""

    window: int = TOPO_WINDOW
    embed_dim: int = TOPO_EMBED_DIM
    homology_dims: Sequence[int] = (0, 1)
    metric: str = "euclidean"


# =============================================================================
# Modeling Configuration
# =============================================================================
DEFAULT_MODEL: Literal["quick", "logistic", "xgboost", "ensemble"] = "quick"
MODEL_CHOICES = ["quick", "logistic", "xgboost", "ensemble"]
TEST_FRACTION = 0.2

# =============================================================================
# Timezone Configuration
# =============================================================================
LOCAL_TZ = "America/Toronto"
MARKET_TZ = "America/New_York"


# =============================================================================
# Configuration Dataclasses (frozen for immutability)
# =============================================================================
@dataclass(frozen=True)
class NewsConfig:
    """Configuration for news fetching."""

    source: Literal["google", "yfinance", "auto"] = NEWS_SOURCE
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    max_headlines_per_ticker: int = MAX_HEADLINES_PER_TICKER


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for ML models."""

    model_type: Literal["quick", "logistic", "xgboost", "ensemble"] = DEFAULT_MODEL
    test_fraction: float = TEST_FRACTION
    calibrate: bool = False
    features: tuple[str, ...] = (
        "ret_1",
        "ret_5",
        "ret_10",
        "ret_20",
        "vol_20",
        "sent_mean",
        "n",
        "topo_d0_total",
        "topo_d1_total",
        "topo_entropy",
    )


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for trading strategy and backtesting."""

    threshold: float = PREDICTION_THRESHOLD
    require_momentum: bool = True
    momentum_col: str = "ret_5"
    cost_bps: float = ROUND_TRIP_COST_BPS
    slippage_bps: float = SLIPPAGE_BPS
    spread_bps: float = SPREAD_BPS
    max_exposure: float = MAX_GROSS_EXPOSURE
    vol_target: float = VOL_TARGET


@dataclass(frozen=True)
class PipelineConfig:
    """Complete configuration for pipeline runs."""

    tickers: tuple[str, ...] = tuple(TICKERS)
    news: NewsConfig = field(default_factory=NewsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    topo: TopoConfig = field(default_factory=TopoConfig)
    enable_topology: bool = ENABLE_TOPOLOGY
    price_lookback_days: int = 180
    output_dir: Path = OUTPUT_DIR


# Default configurations
DEFAULT_NEWS_CONFIG = NewsConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_STRATEGY_CONFIG = StrategyConfig()
DEFAULT_TOPO_CONFIG = TopoConfig()
DEFAULT_PIPELINE_CONFIG = PipelineConfig()
