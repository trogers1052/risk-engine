"""Tests for risk calculators."""

import numpy as np
import pandas as pd
import pytest

from risk_engine.calculators.position_sizer import PositionSizer
from risk_engine.calculators.metrics import MetricsCalculator
from risk_engine.config import RiskSettings


@pytest.fixture
def settings():
    """Create test settings."""
    return RiskSettings()


@pytest.fixture
def sample_returns():
    """Create sample return series."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.02, 252))
    return returns


class TestPositionSizer:
    """Tests for PositionSizer."""

    def test_atr_sizing(self, settings):
        sizer = PositionSizer(settings)

        result = sizer.calculate(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100000.0,
            buying_power=50000.0,
            atr=3.0,
        )

        assert result.shares > 0
        assert result.method == "atr"
        assert result.dollar_amount <= result.max_dollar_amount

    def test_atr_sizing_respects_max_position(self, settings):
        sizer = PositionSizer(settings)

        # Very low ATR would normally give huge position
        result = sizer.calculate(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100000.0,
            buying_power=50000.0,
            atr=0.1,  # Very low ATR
        )

        # Should be capped at max position (20% by default)
        max_dollar = 100000.0 * 0.20
        assert result.dollar_amount <= max_dollar

    def test_buying_power_limit(self, settings):
        sizer = PositionSizer(settings)

        result = sizer.calculate(
            symbol="AAPL",
            price=150.0,
            portfolio_value=100000.0,
            buying_power=1000.0,  # Very limited
            atr=3.0,
        )

        # Should be limited by buying power
        assert result.dollar_amount <= 1000.0
        assert "buying power" in result.notes.lower()

    def test_stop_loss_calculation(self, settings):
        sizer = PositionSizer(settings)

        stop = sizer.calculate_stop_loss(
            entry_price=100.0,
            atr=2.0,
            multiplier=2.0,
        )

        assert stop == 96.0  # 100 - (2 * 2)

    def test_position_for_risk(self, settings):
        sizer = PositionSizer(settings)

        shares = sizer.calculate_position_for_risk(
            entry_price=100.0,
            stop_price=95.0,
            risk_amount=500.0,
        )

        # Risk per share = 5, risk amount = 500, so 100 shares
        assert shares == 100


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_calculate_metrics(self, sample_returns):
        calc = MetricsCalculator()
        metrics = calc.calculate_metrics(sample_returns)

        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "sortino_ratio")
        assert hasattr(metrics, "max_drawdown")
        assert hasattr(metrics, "annual_volatility")

    def test_sharpe_ratio(self, sample_returns):
        calc = MetricsCalculator()
        sharpe = calc.calculate_sharpe(sample_returns)

        # With our random seed, should be a reasonable value
        assert -5 < sharpe < 5

    def test_max_drawdown(self, sample_returns):
        calc = MetricsCalculator()
        max_dd = calc.calculate_max_drawdown(sample_returns)

        # Should be positive (magnitude)
        assert max_dd >= 0
        assert max_dd <= 1  # Can't lose more than 100%

    def test_rolling_volatility(self, sample_returns):
        calc = MetricsCalculator()
        rolling_vol = calc.calculate_rolling_volatility(sample_returns, window=21)

        # Should have NaN for first window-1 values
        assert pd.isna(rolling_vol.iloc[0])
        assert not pd.isna(rolling_vol.iloc[-1])

    def test_var_calculation(self, sample_returns):
        calc = MetricsCalculator()
        var = calc._calculate_var(sample_returns, 0.95)

        # VaR should be positive
        assert var > 0

    def test_cvar_calculation(self, sample_returns):
        calc = MetricsCalculator()
        var = calc._calculate_var(sample_returns, 0.95)
        cvar = calc._calculate_cvar(sample_returns, 0.95)

        # CVaR should be >= VaR
        assert cvar >= var

    def test_metrics_to_dict(self, sample_returns):
        calc = MetricsCalculator()
        metrics = calc.calculate_metrics(sample_returns)
        d = metrics.to_dict()

        assert "sharpe_ratio" in d
        assert "sortino_ratio" in d
        assert "max_drawdown" in d
