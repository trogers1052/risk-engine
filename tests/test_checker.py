"""Tests for RiskChecker orchestrator."""

import pytest
from unittest.mock import MagicMock, patch

from risk_engine.checker import RiskChecker
from risk_engine.config import RiskSettings
from risk_engine.models.portfolio import Position, PortfolioState


@pytest.fixture
def settings():
    """Create test settings."""
    return RiskSettings()


@pytest.fixture
def portfolio():
    """Create test portfolio."""
    pos = Position(
        symbol="AAPL",
        shares=100,
        average_cost=150.0,
        current_price=160.0,
        market_value=16000.0,
        unrealized_pnl=1000.0,
        unrealized_pnl_pct=0.0667,
    )
    return PortfolioState(
        positions={"AAPL": pos},
        buying_power=10000.0,
        cash=10000.0,
        total_equity=26000.0,
        market_value=16000.0,
    )


@pytest.fixture
def indicators():
    """Create test indicators."""
    return {
        "close": 350.0,
        "atr_14": 7.0,
        "sma_200": 320.0,
        "rsi_14": 55.0,
        "volume": 1000000,
        "avg_volume_20": 800000,
    }


class TestRiskChecker:
    """Tests for RiskChecker."""

    def test_initialization(self, settings):
        checker = RiskChecker(settings)
        assert len(checker._checks) >= 3  # At least trade, portfolio, position sizing

    def test_check_buy_passes(self, settings, portfolio, indicators):
        # Mock portfolio manager to return our test portfolio
        mock_portfolio_manager = MagicMock()
        mock_portfolio_manager.get_portfolio_state.return_value = portfolio

        checker = RiskChecker(
            settings=settings,
            portfolio_manager=mock_portfolio_manager,
        )

        result = checker.check(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators=indicators,
        )

        assert result.passes is True
        assert result.recommended_shares > 0

    def test_check_high_volatility_fails(self, settings, portfolio):
        mock_portfolio_manager = MagicMock()
        mock_portfolio_manager.get_portfolio_state.return_value = portfolio

        checker = RiskChecker(
            settings=settings,
            portfolio_manager=mock_portfolio_manager,
        )

        # High volatility indicators
        high_vol_indicators = {
            "close": 100.0,
            "atr_14": 15.0,  # 15% volatility
            "sma_200": 90.0,
        }

        result = checker.check(
            symbol="MEME",
            signal_type="BUY",
            confidence=0.8,
            indicators=high_vol_indicators,
        )

        assert result.passes is False
        assert "volatility" in result.reason.lower()

    def test_check_below_sma_fails(self, settings, portfolio):
        mock_portfolio_manager = MagicMock()
        mock_portfolio_manager.get_portfolio_state.return_value = portfolio

        checker = RiskChecker(
            settings=settings,
            portfolio_manager=mock_portfolio_manager,
        )

        # Price below SMA 200
        below_sma_indicators = {
            "close": 100.0,
            "atr_14": 2.0,
            "sma_200": 120.0,  # Price below SMA
        }

        result = checker.check(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators=below_sma_indicators,
        )

        assert result.passes is False
        assert "sma" in result.reason.lower()

    def test_check_convenience_methods(self, settings, portfolio, indicators):
        mock_portfolio_manager = MagicMock()
        mock_portfolio_manager.get_portfolio_state.return_value = portfolio

        checker = RiskChecker(
            settings=settings,
            portfolio_manager=mock_portfolio_manager,
        )

        # Test check_buy
        result = checker.check_buy(
            symbol="MSFT",
            confidence=0.8,
            indicators=indicators,
        )
        assert result.passes is True

        # Test check_sell (should always pass)
        result = checker.check_sell(
            symbol="MSFT",
            confidence=0.8,
            indicators=indicators,
        )
        assert result.passes is True

    def test_risk_metrics_aggregation(self, settings, portfolio, indicators):
        mock_portfolio_manager = MagicMock()
        mock_portfolio_manager.get_portfolio_state.return_value = portfolio

        checker = RiskChecker(
            settings=settings,
            portfolio_manager=mock_portfolio_manager,
        )

        result = checker.check_buy(
            symbol="MSFT",
            confidence=0.8,
            indicators=indicators,
        )

        # Should have aggregated metrics from multiple checks
        assert len(result.risk_metrics) > 0
