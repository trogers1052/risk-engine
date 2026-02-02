"""Tests for stop loss calculator."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from risk_engine.calculators.stop_loss import StopLossCalculator, StopLossCalculation
from risk_engine.models.portfolio import Position, PortfolioState
from risk_engine.config import RiskSettings


@pytest.fixture
def settings():
    """Create test settings."""
    return RiskSettings()


@pytest.fixture
def mock_portfolio_manager(settings):
    """Create mock portfolio manager."""
    manager = MagicMock()

    # Create test positions
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            shares=100,
            average_cost=150.00,
            current_price=155.00,
            market_value=15500.00,
            unrealized_pnl=500.00,
            unrealized_pnl_pct=3.33,
        ),
        "GOOGL": Position(
            symbol="GOOGL",
            shares=50,
            average_cost=140.00,
            current_price=133.00,
            market_value=6650.00,
            unrealized_pnl=-350.00,
            unrealized_pnl_pct=-5.0,
        ),
    }

    portfolio = PortfolioState(
        positions=positions,
        buying_power=5000.00,
        total_equity=27150.00,
    )

    manager.get_portfolio_state.return_value = portfolio
    return manager


@pytest.fixture
def mock_rules_client():
    """Create mock rules client."""
    client = MagicMock()

    # Default exit strategy
    client.get_exit_strategy.return_value = {
        "profit_target": 0.07,
        "stop_loss": 0.05,
    }

    return client


@pytest.fixture
def stop_loss_calculator(settings, mock_portfolio_manager, mock_rules_client):
    """Create stop loss calculator."""
    return StopLossCalculator(
        settings=settings,
        portfolio_manager=mock_portfolio_manager,
        rules_client=mock_rules_client,
    )


class TestStopLossCalculation:
    """Tests for StopLossCalculation dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        calc = StopLossCalculation(
            symbol="AAPL",
            shares=100,
            average_cost=150.00,
            current_price=155.00,
            stop_loss_pct=0.05,
            stop_loss_price=142.50,
            stop_loss_dollar_amount=14250.00,
            potential_loss=750.00,
            potential_loss_pct=0.05,
            distance_to_stop=12.50,
            distance_to_stop_pct=0.0806,
            is_triggered=False,
            profit_target_pct=0.07,
            profit_target_price=160.50,
        )

        d = calc.to_dict()

        assert d["symbol"] == "AAPL"
        assert d["shares"] == 100
        assert d["stop_loss_price"] == 142.50
        assert d["is_triggered"] is False

    def test_format_alert_message(self):
        """Test alert message formatting."""
        calc = StopLossCalculation(
            symbol="AAPL",
            shares=100,
            average_cost=150.00,
            current_price=155.00,
            stop_loss_pct=0.05,
            stop_loss_price=142.50,
            stop_loss_dollar_amount=14250.00,
            potential_loss=750.00,
            potential_loss_pct=0.05,
            distance_to_stop=12.50,
            distance_to_stop_pct=0.0806,
            is_triggered=False,
            profit_target_pct=0.07,
            profit_target_price=160.50,
        )

        message = calc.format_alert_message()

        assert "AAPL" in message
        assert "$142.50" in message
        assert "100" in message

    def test_triggered_alert_message(self):
        """Test alert message when stop loss is triggered."""
        calc = StopLossCalculation(
            symbol="GOOGL",
            shares=50,
            average_cost=140.00,
            current_price=130.00,
            stop_loss_pct=0.05,
            stop_loss_price=133.00,
            stop_loss_dollar_amount=6650.00,
            potential_loss=350.00,
            potential_loss_pct=0.05,
            distance_to_stop=-3.00,
            distance_to_stop_pct=-0.023,
            is_triggered=True,
            profit_target_pct=0.07,
            profit_target_price=149.80,
        )

        message = calc.format_alert_message()

        assert "TRIGGERED" in message
        assert "GOOGL" in message


class TestStopLossCalculator:
    """Tests for StopLossCalculator."""

    def test_calculate_for_position(self, stop_loss_calculator, mock_portfolio_manager):
        """Test stop loss calculation for a position."""
        portfolio = mock_portfolio_manager.get_portfolio_state()
        position = portfolio.get_position("AAPL")

        calc = stop_loss_calculator.calculate_for_position(position)

        assert calc.symbol == "AAPL"
        assert calc.shares == 100
        assert calc.average_cost == 150.00
        assert calc.stop_loss_pct == 0.05
        assert calc.stop_loss_price == 142.50  # 150 * (1 - 0.05)
        assert calc.stop_loss_dollar_amount == 14250.00  # 142.50 * 100
        assert calc.is_triggered is False

    def test_calculate_with_symbol_override(self, settings, mock_portfolio_manager):
        """Test stop loss with symbol-specific override."""
        rules_client = MagicMock()
        # Higher stop loss for volatile stock
        rules_client.get_exit_strategy.return_value = {
            "profit_target": 0.10,
            "stop_loss": 0.08,
        }

        calculator = StopLossCalculator(
            settings=settings,
            portfolio_manager=mock_portfolio_manager,
            rules_client=rules_client,
        )

        portfolio = mock_portfolio_manager.get_portfolio_state()
        position = portfolio.get_position("AAPL")

        calc = calculator.calculate_for_position(position)

        assert calc.stop_loss_pct == 0.08
        assert calc.stop_loss_price == 138.00  # 150 * (1 - 0.08)
        assert calc.profit_target_pct == 0.10
        assert calc.profit_target_price == 165.00  # 150 * (1 + 0.10)

    def test_calculate_triggered_stop(self, settings, mock_portfolio_manager, mock_rules_client):
        """Test calculation when stop loss is triggered."""
        calculator = StopLossCalculator(
            settings=settings,
            portfolio_manager=mock_portfolio_manager,
            rules_client=mock_rules_client,
        )

        portfolio = mock_portfolio_manager.get_portfolio_state()
        position = portfolio.get_position("GOOGL")

        calc = calculator.calculate_for_position(position)

        # GOOGL is at $133, stop loss is $133 (140 * 0.95)
        assert calc.stop_loss_price == 133.00
        assert calc.is_triggered is True

    def test_calculate_no_position(self, stop_loss_calculator):
        """Test calculation when no position exists."""
        result = stop_loss_calculator.calculate("UNKNOWN")
        assert result is None

    def test_calculate_from_entry(self, stop_loss_calculator):
        """Test calculation from entry parameters."""
        calc = stop_loss_calculator.calculate_from_entry(
            symbol="MSFT",
            entry_price=400.00,
            shares=25,
            current_price=405.00,
        )

        assert calc.symbol == "MSFT"
        assert calc.shares == 25
        assert calc.average_cost == 400.00
        assert calc.current_price == 405.00
        assert calc.stop_loss_price == 380.00  # 400 * (1 - 0.05)
        assert calc.stop_loss_dollar_amount == 9500.00  # 380 * 25
        assert calc.is_triggered is False

    def test_calculate_all(self, stop_loss_calculator):
        """Test calculating stop losses for all positions."""
        calcs = stop_loss_calculator.calculate_all()

        assert len(calcs) == 2
        symbols = {c.symbol for c in calcs}
        assert "AAPL" in symbols
        assert "GOOGL" in symbols

    def test_get_triggered_stops(self, stop_loss_calculator):
        """Test getting only triggered stops."""
        triggered = stop_loss_calculator.get_triggered_stops()

        # Only GOOGL should be triggered
        assert len(triggered) == 1
        assert triggered[0].symbol == "GOOGL"
