"""Pytest configuration and shared fixtures."""

import numpy as np
import pandas as pd
import pytest

from risk_engine.config import RiskSettings
from risk_engine.models.portfolio import Position, PortfolioState


@pytest.fixture
def settings():
    """Create default test settings."""
    return RiskSettings()


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            shares=100,
            average_cost=150.0,
            current_price=160.0,
            market_value=16000.0,
            unrealized_pnl=1000.0,
            unrealized_pnl_pct=0.0667,
        ),
        "GOOGL": Position(
            symbol="GOOGL",
            shares=50,
            average_cost=100.0,
            current_price=110.0,
            market_value=5500.0,
            unrealized_pnl=500.0,
            unrealized_pnl_pct=0.10,
        ),
    }
    return PortfolioState(
        positions=positions,
        buying_power=10000.0,
        cash=10000.0,
        total_equity=31500.0,
        market_value=21500.0,
    )


@pytest.fixture
def empty_portfolio():
    """Create an empty portfolio for testing."""
    return PortfolioState(
        buying_power=50000.0,
        cash=50000.0,
        total_equity=50000.0,
    )


@pytest.fixture
def sample_indicators():
    """Create sample indicator values."""
    return {
        "close": 150.0,
        "open": 148.0,
        "high": 152.0,
        "low": 147.0,
        "volume": 1000000,
        "atr_14": 3.0,
        "rsi_14": 55.0,
        "sma_20": 148.0,
        "sma_50": 145.0,
        "sma_200": 140.0,
        "ema_12": 149.0,
        "ema_26": 147.0,
        "macd": 2.0,
        "macd_signal": 1.5,
        "macd_histogram": 0.5,
        "avg_volume_20": 900000,
    }


@pytest.fixture
def sample_returns():
    """Create sample daily returns series."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    returns = pd.Series(
        np.random.normal(0.0005, 0.02, 252),
        index=dates,
        name="returns",
    )
    return returns


@pytest.fixture
def multi_asset_returns():
    """Create multi-asset returns DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")

    # Create correlated returns
    n = 252
    cov = np.array([
        [0.0004, 0.0002, 0.0001],
        [0.0002, 0.0006, 0.0002],
        [0.0001, 0.0002, 0.0003],
    ])
    mean = [0.0005, 0.0003, 0.0004]
    returns = np.random.multivariate_normal(mean, cov, n)

    df = pd.DataFrame(
        returns,
        index=dates,
        columns=["AAPL", "GOOGL", "MSFT"],
    )
    return df


@pytest.fixture
def high_volatility_indicators():
    """Create indicators for a high-volatility stock."""
    return {
        "close": 100.0,
        "atr_14": 10.0,  # 10% volatility
        "sma_200": 95.0,
        "volume": 500000,
        "avg_volume_20": 400000,
    }


@pytest.fixture
def low_liquidity_indicators():
    """Create indicators for a low-liquidity situation."""
    return {
        "close": 150.0,
        "atr_14": 3.0,
        "sma_200": 140.0,
        "volume": 100000,
        "avg_volume_20": 500000,  # Current volume is 20% of average
    }


@pytest.fixture
def bearish_indicators():
    """Create indicators for a bearish stock (below SMA 200)."""
    return {
        "close": 100.0,
        "atr_14": 2.0,
        "sma_200": 120.0,  # Price 17% below SMA 200
        "volume": 1000000,
        "avg_volume_20": 800000,
    }
