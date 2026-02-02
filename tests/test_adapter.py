"""Tests for RiskAdapter integration."""

import pytest
from unittest.mock import MagicMock, patch

from risk_engine.adapter import RiskAdapter, get_risk_adapter
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


class TestRiskAdapter:
    """Tests for RiskAdapter."""

    def test_initialization(self, settings):
        adapter = RiskAdapter(settings=settings)
        assert adapter._initialized is False

    @patch("risk_engine.adapter.AlertPublisher")
    @patch("risk_engine.adapter.RulesClient")
    @patch("risk_engine.adapter.PortfolioStateManager")
    @patch("risk_engine.adapter.MarketDataLoader")
    def test_initialize(self, mock_market_data, mock_portfolio_manager, mock_rules_client, mock_alert_publisher, settings):
        # Setup mocks
        mock_portfolio_manager.return_value.connect.return_value = True
        mock_market_data.return_value.connect.return_value = True
        mock_rules_client.return_value.connect.return_value = True
        mock_alert_publisher.return_value.connect.return_value = True

        adapter = RiskAdapter(settings=settings)
        result = adapter.initialize()

        assert result is True
        assert adapter._initialized is True

    @patch("risk_engine.adapter.AlertPublisher")
    @patch("risk_engine.adapter.RulesClient")
    @patch("risk_engine.adapter.PortfolioStateManager")
    @patch("risk_engine.adapter.MarketDataLoader")
    def test_check_risk_buy(
        self, mock_market_data, mock_portfolio_manager, mock_rules_client, mock_alert_publisher, settings, portfolio
    ):
        # Setup mocks
        mock_pm_instance = MagicMock()
        mock_pm_instance.connect.return_value = True
        mock_pm_instance.get_portfolio_state.return_value = portfolio
        mock_portfolio_manager.return_value = mock_pm_instance

        mock_md_instance = MagicMock()
        mock_md_instance.connect.return_value = True
        mock_md_instance.get_max_correlation_with_portfolio.return_value = (None, 0.0)
        mock_market_data.return_value = mock_md_instance

        mock_rules_client.return_value.connect.return_value = True
        mock_alert_publisher.return_value.connect.return_value = True

        adapter = RiskAdapter(settings=settings)
        adapter.initialize()

        result = adapter.check_risk(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={
                "close": 350.0,
                "atr_14": 7.0,
                "sma_200": 320.0,
                "volume": 1000000,
                "avg_volume_20": 800000,
            },
        )

        assert result is not None
        assert hasattr(result, "passes")

    def test_check_risk_sell_passthrough(self, settings):
        adapter = RiskAdapter(settings=settings)

        # SELL signals should pass through without checks
        result = adapter.check_risk(
            symbol="MSFT",
            signal_type="SELL",
            confidence=0.8,
            indicators={"close": 350.0},
        )

        assert result.passes is True
        assert "not risk-checked" in result.reason

    def test_check_risk_watch_passthrough(self, settings):
        adapter = RiskAdapter(settings=settings)

        # WATCH signals should pass through without checks
        result = adapter.check_risk(
            symbol="MSFT",
            signal_type="WATCH",
            confidence=0.5,
            indicators={"close": 350.0},
        )

        assert result.passes is True

    @patch("risk_engine.adapter.AlertPublisher")
    @patch("risk_engine.adapter.RulesClient")
    @patch("risk_engine.adapter.PortfolioStateManager")
    @patch("risk_engine.adapter.MarketDataLoader")
    def test_get_portfolio_summary(
        self, mock_market_data, mock_portfolio_manager, mock_rules_client, mock_alert_publisher, settings, portfolio
    ):
        # Setup mocks
        mock_pm_instance = MagicMock()
        mock_pm_instance.connect.return_value = True
        mock_pm_instance.get_portfolio_state.return_value = portfolio
        mock_portfolio_manager.return_value = mock_pm_instance

        mock_md_instance = MagicMock()
        mock_md_instance.connect.return_value = True
        mock_market_data.return_value = mock_md_instance

        mock_rules_client.return_value.connect.return_value = True
        mock_alert_publisher.return_value.connect.return_value = True

        adapter = RiskAdapter(settings=settings)
        adapter.initialize()

        summary = adapter.get_portfolio_summary()

        assert "positions" in summary
        assert "total_equity" in summary

    @patch("risk_engine.adapter.AlertPublisher")
    @patch("risk_engine.adapter.RulesClient")
    @patch("risk_engine.adapter.PortfolioStateManager")
    @patch("risk_engine.adapter.MarketDataLoader")
    def test_shutdown(self, mock_market_data, mock_portfolio_manager, mock_rules_client, mock_alert_publisher, settings):
        mock_pm_instance = MagicMock()
        mock_pm_instance.connect.return_value = True
        mock_portfolio_manager.return_value = mock_pm_instance

        mock_md_instance = MagicMock()
        mock_md_instance.connect.return_value = True
        mock_market_data.return_value = mock_md_instance

        mock_rules_instance = MagicMock()
        mock_rules_instance.connect.return_value = True
        mock_rules_client.return_value = mock_rules_instance

        mock_alert_instance = MagicMock()
        mock_alert_instance.connect.return_value = True
        mock_alert_publisher.return_value = mock_alert_instance

        adapter = RiskAdapter(settings=settings)
        adapter.initialize()
        adapter.shutdown()

        assert adapter._initialized is False
        mock_pm_instance.close.assert_called_once()
        mock_md_instance.close.assert_called_once()


class TestGetRiskAdapter:
    """Tests for get_risk_adapter singleton."""

    def test_singleton(self, settings):
        # Reset singleton for test
        import risk_engine.adapter

        risk_engine.adapter._default_adapter = None

        adapter1 = get_risk_adapter(settings=settings)
        adapter2 = get_risk_adapter()

        assert adapter1 is adapter2
