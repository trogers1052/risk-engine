"""Tests for risk engine models."""

import pytest
from risk_engine.models.risk_result import RiskCheckResult, RiskLevel
from risk_engine.models.portfolio import Position, PortfolioState


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_from_score_low(self):
        assert RiskLevel.from_score(0.10) == RiskLevel.LOW
        assert RiskLevel.from_score(0.24) == RiskLevel.LOW

    def test_from_score_medium(self):
        assert RiskLevel.from_score(0.25) == RiskLevel.MEDIUM
        assert RiskLevel.from_score(0.49) == RiskLevel.MEDIUM

    def test_from_score_high(self):
        assert RiskLevel.from_score(0.50) == RiskLevel.HIGH
        assert RiskLevel.from_score(0.74) == RiskLevel.HIGH

    def test_from_score_critical(self):
        assert RiskLevel.from_score(0.75) == RiskLevel.CRITICAL
        assert RiskLevel.from_score(1.0) == RiskLevel.CRITICAL


class TestRiskCheckResult:
    """Tests for RiskCheckResult dataclass."""

    def test_reject(self):
        result = RiskCheckResult.reject(
            reason="Test rejection",
            risk_score=0.9,
            symbol="AAPL",
        )
        assert result.passes is False
        assert result.reason == "Test rejection"
        assert result.risk_score == 0.9
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.recommended_shares == 0
        assert result.symbol == "AAPL"

    def test_approve(self):
        result = RiskCheckResult.approve(
            recommended_shares=100,
            max_shares=150,
            recommended_dollar_amount=15000.0,
            risk_score=0.3,
            symbol="AAPL",
        )
        assert result.passes is True
        assert result.recommended_shares == 100
        assert result.max_shares == 150
        assert result.recommended_dollar_amount == 15000.0
        assert result.risk_level == RiskLevel.MEDIUM

    def test_to_dict(self):
        result = RiskCheckResult.approve(
            recommended_shares=100,
            max_shares=150,
            recommended_dollar_amount=15000.0,
            risk_score=0.3,
            symbol="AAPL",
            risk_metrics={"var_95": 0.045},
        )
        d = result.to_dict()
        assert d["passes"] is True
        assert d["risk_metrics"]["var_95"] == 0.045


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        pos = Position(
            symbol="AAPL",
            shares=100,
            average_cost=150.0,
            current_price=160.0,
            market_value=16000.0,
            unrealized_pnl=1000.0,
            unrealized_pnl_pct=0.0667,
        )
        assert pos.symbol == "AAPL"
        assert pos.cost_basis == 15000.0

    def test_position_to_dict(self):
        pos = Position(
            symbol="AAPL",
            shares=100,
            average_cost=150.0,
            current_price=160.0,
            market_value=16000.0,
            unrealized_pnl=1000.0,
            unrealized_pnl_pct=0.0667,
        )
        d = pos.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["cost_basis"] == 15000.0


class TestPortfolioState:
    """Tests for PortfolioState dataclass."""

    def test_empty_portfolio(self):
        state = PortfolioState()
        assert state.position_count == 0
        assert state.total_positions_value == 0
        assert state.cash_pct == 1.0

    def test_portfolio_with_positions(self):
        pos1 = Position(
            symbol="AAPL",
            shares=100,
            average_cost=150.0,
            current_price=160.0,
            market_value=16000.0,
            unrealized_pnl=1000.0,
            unrealized_pnl_pct=0.0667,
        )
        pos2 = Position(
            symbol="GOOGL",
            shares=50,
            average_cost=100.0,
            current_price=110.0,
            market_value=5500.0,
            unrealized_pnl=500.0,
            unrealized_pnl_pct=0.10,
        )

        state = PortfolioState(
            positions={"AAPL": pos1, "GOOGL": pos2},
            total_equity=25000.0,
            cash=3500.0,
        )

        assert state.position_count == 2
        assert state.total_positions_value == 21500.0
        assert state.get_position_weight("AAPL") == pytest.approx(0.64, rel=0.01)
        assert state.has_position("AAPL") is True
        assert state.has_position("MSFT") is False

    def test_get_weights(self):
        pos = Position(
            symbol="AAPL",
            shares=100,
            average_cost=150.0,
            current_price=160.0,
            market_value=10000.0,
            unrealized_pnl=1000.0,
            unrealized_pnl_pct=0.0667,
        )
        state = PortfolioState(
            positions={"AAPL": pos},
            total_equity=20000.0,
        )
        weights = state.get_weights()
        assert weights["AAPL"] == 0.5
