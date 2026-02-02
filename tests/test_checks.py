"""Tests for risk checks."""

import pytest
from risk_engine.checks.base import RiskCheckContext
from risk_engine.checks.position_sizing import PositionSizingCheck
from risk_engine.checks.portfolio_risk import PortfolioRiskCheck
from risk_engine.checks.trade_risk import TradeRiskCheck
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


class TestPositionSizingCheck:
    """Tests for PositionSizingCheck."""

    def test_atr_sizing(self, settings, portfolio):
        check = PositionSizingCheck(settings)

        context = RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 350.0, "atr_14": 7.0},
            portfolio=portfolio,
            current_price=350.0,
        )

        result = check.check(context)

        assert result.passes is True
        assert result.recommended_shares > 0
        assert result.recommended_dollar_amount > 0

    def test_no_price_rejection(self, settings, portfolio):
        check = PositionSizingCheck(settings)

        context = RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={"atr_14": 7.0},
            portfolio=portfolio,
            current_price=None,
        )

        result = check.check(context)
        assert result.passes is False

    def test_insufficient_buying_power(self, settings):
        check = PositionSizingCheck(settings)

        # Portfolio with very low buying power
        small_portfolio = PortfolioState(
            buying_power=10.0,
            cash=10.0,
            total_equity=1000.0,
        )

        context = RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 350.0, "atr_14": 7.0},
            portfolio=small_portfolio,
            current_price=350.0,
        )

        result = check.check(context)
        assert result.passes is False
        assert "buying power" in result.reason.lower()


class TestTradeRiskCheck:
    """Tests for TradeRiskCheck."""

    def test_high_volatility_rejection(self, settings, portfolio):
        check = TradeRiskCheck(settings)

        # ATR is 10% of price - exceeds 5% limit
        context = RiskCheckContext(
            symbol="MEME",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0, "atr_14": 10.0},
            portfolio=portfolio,
            current_price=100.0,
        )

        result = check.check(context)
        assert result.passes is False
        assert "volatility" in result.reason.lower()

    def test_below_sma200_rejection(self, settings, portfolio):
        check = TradeRiskCheck(settings)

        # Price below SMA 200
        context = RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={
                "close": 100.0,
                "atr_14": 2.0,
                "sma_200": 120.0,
            },
            portfolio=portfolio,
            current_price=100.0,
        )

        result = check.check(context)
        assert result.passes is False
        assert "sma" in result.reason.lower()

    def test_acceptable_trade(self, settings, portfolio):
        check = TradeRiskCheck(settings)

        # Good conditions: low volatility, above SMA
        context = RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={
                "close": 150.0,
                "atr_14": 3.0,  # 2% volatility
                "sma_200": 140.0,
                "volume": 1000000,
                "avg_volume_20": 800000,
            },
            portfolio=portfolio,
            current_price=150.0,
        )

        result = check.check(context)
        assert result.passes is True


class TestPortfolioRiskCheck:
    """Tests for PortfolioRiskCheck."""

    def test_concentration_limit(self, settings):
        check = PortfolioRiskCheck(settings)

        # Position already at 25% concentration
        large_pos = Position(
            symbol="AAPL",
            shares=100,
            average_cost=150.0,
            current_price=250.0,
            market_value=25000.0,
            unrealized_pnl=10000.0,
            unrealized_pnl_pct=0.67,
        )
        concentrated_portfolio = PortfolioState(
            positions={"AAPL": large_pos},
            total_equity=100000.0,
            cash=75000.0,
        )

        context = RiskCheckContext(
            symbol="AAPL",  # Trying to add to existing position
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 250.0},
            portfolio=concentrated_portfolio,
            current_price=250.0,
        )

        result = check.check(context)
        assert result.passes is False
        assert "concentration" in result.reason.lower()

    def test_exposure_limit(self, settings):
        check = PortfolioRiskCheck(settings)

        # Portfolio at 96% exposure (over 95% limit)
        pos = Position(
            symbol="AAPL",
            shares=960,
            average_cost=100.0,
            current_price=100.0,
            market_value=96000.0,
            unrealized_pnl=0,
            unrealized_pnl_pct=0,
        )
        maxed_portfolio = PortfolioState(
            positions={"AAPL": pos},
            total_equity=100000.0,
            cash=4000.0,
            market_value=96000.0,
        )

        context = RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 350.0},
            portfolio=maxed_portfolio,
            current_price=350.0,
        )

        result = check.check(context)
        assert result.passes is False
        assert "exposure" in result.reason.lower()

    def test_acceptable_portfolio(self, settings, portfolio):
        check = PortfolioRiskCheck(settings)

        context = RiskCheckContext(
            symbol="MSFT",  # New position
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 350.0},
            portfolio=portfolio,
            current_price=350.0,
        )

        result = check.check(context)
        assert result.passes is True
