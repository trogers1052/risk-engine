"""Tests for risk checks."""

import pytest
from unittest.mock import MagicMock
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

        # Portfolio with enough equity for position size but low buying power
        # Equity of 10000 allows max 2000 position (20%), price 100 = 20 shares
        # But buying power of 50 means we can't afford even 1 share at $100
        small_portfolio = PortfolioState(
            buying_power=50.0,
            cash=50.0,
            total_equity=10000.0,
        )

        context = RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0, "atr_14": 2.0},
            portfolio=small_portfolio,
            current_price=100.0,
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


class TestOpenPositionsCheck:
    """Tests for max open positions check."""

    def test_rejects_when_at_max_positions(self, settings):
        """Should reject when position count >= max_open_positions."""
        # Create portfolio with 5 positions (at the default max of 5)
        positions = {}
        for i, sym in enumerate(["AAPL", "GOOGL", "MSFT", "AMZN", "META"]):
            positions[sym] = Position(
                symbol=sym,
                shares=10,
                average_cost=100.0,
                current_price=110.0,
                market_value=1100.0,
                unrealized_pnl=100.0,
                unrealized_pnl_pct=0.10,
            )
        full_portfolio = PortfolioState(
            positions=positions,
            buying_power=50000.0,
            cash=50000.0,
            total_equity=100000.0,
            market_value=5500.0,
        )

        check = PortfolioRiskCheck(settings)
        context = RiskCheckContext(
            symbol="TSLA",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 200.0},
            portfolio=full_portfolio,
            current_price=200.0,
        )

        result = check.check(context)
        assert result.passes is False
        assert "max positions" in result.reason.lower()

    def test_passes_under_max_positions(self, settings, portfolio):
        """Should pass when position count is below max."""
        check = PortfolioRiskCheck(settings)

        context = RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 350.0},
            portfolio=portfolio,
            current_price=350.0,
        )

        result = check.check(context)
        assert result.passes is True

    def test_warns_near_position_limit(self, settings):
        """Should warn when one away from max positions."""
        # Create portfolio with 4 positions (max - 1)
        positions = {}
        for sym in ["AAPL", "GOOGL", "MSFT", "AMZN"]:
            positions[sym] = Position(
                symbol=sym,
                shares=10,
                average_cost=100.0,
                current_price=110.0,
                market_value=1100.0,
                unrealized_pnl=100.0,
                unrealized_pnl_pct=0.10,
            )
        near_full = PortfolioState(
            positions=positions,
            buying_power=50000.0,
            cash=50000.0,
            total_equity=100000.0,
            market_value=4400.0,
        )

        check = PortfolioRiskCheck(settings)
        context = RiskCheckContext(
            symbol="TSLA",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 200.0},
            portfolio=near_full,
            current_price=200.0,
        )

        result = check.check(context)
        assert result.passes is True
        assert any("position limit" in w.lower() for w in result.warnings)


class TestSectorConcentrationCheck:
    """Tests for sector concentration check."""

    def _settings_with_sectors(self):
        """Create settings with sector groups."""
        return RiskSettings(
            sector_groups={
                "uranium": ["CCJ", "URNM", "UUUU"],
                "precious_metals": ["WPM", "SLV", "IAUM", "PPLT"],
            }
        )

    def test_rejects_when_sector_at_limit(self):
        """Should reject when sector exposure >= max_sector_exposure_pct."""
        settings = self._settings_with_sectors()

        # CCJ and URNM each at 21% = 42% sector exposure (above 40%)
        positions = {
            "CCJ": Position(
                symbol="CCJ",
                shares=100,
                average_cost=50.0,
                current_price=50.0,
                market_value=5000.0,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
            ),
            "URNM": Position(
                symbol="URNM",
                shares=200,
                average_cost=50.0,
                current_price=50.0,
                market_value=10000.0,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
            ),
        }
        portfolio = PortfolioState(
            positions=positions,
            buying_power=20000.0,
            cash=20000.0,
            total_equity=35000.0,
            market_value=15000.0,
        )

        check = PortfolioRiskCheck(settings)
        context = RiskCheckContext(
            symbol="UUUU",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 10.0},
            portfolio=portfolio,
            current_price=10.0,
        )

        result = check.check(context)
        assert result.passes is False
        assert "sector" in result.reason.lower()
        assert "uranium" in result.reason.lower()

    def test_passes_when_sector_under_limit(self):
        """Should pass when sector exposure is below limit."""
        settings = self._settings_with_sectors()

        # Only CCJ at 15% — adding URNM would keep uranium under 40%
        positions = {
            "CCJ": Position(
                symbol="CCJ",
                shares=100,
                average_cost=50.0,
                current_price=50.0,
                market_value=5000.0,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
            ),
        }
        portfolio = PortfolioState(
            positions=positions,
            buying_power=30000.0,
            cash=30000.0,
            total_equity=35000.0,
            market_value=5000.0,
        )

        check = PortfolioRiskCheck(settings)
        context = RiskCheckContext(
            symbol="URNM",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 50.0},
            portfolio=portfolio,
            current_price=50.0,
        )

        result = check.check(context)
        assert result.passes is True

    def test_unknown_sector_passes_with_warning(self):
        """Symbol not in any sector group passes with warning."""
        settings = self._settings_with_sectors()

        portfolio = PortfolioState(
            buying_power=50000.0,
            cash=50000.0,
            total_equity=50000.0,
        )

        check = PortfolioRiskCheck(settings)
        context = RiskCheckContext(
            symbol="XYZ",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0},
            portfolio=portfolio,
            current_price=100.0,
        )

        result = check.check(context)
        assert result.passes is True
        assert any("no sector mapping" in w.lower() for w in result.warnings)

    def test_warns_near_sector_limit(self):
        """Should warn when sector is close to limit."""
        settings = self._settings_with_sectors()

        # CCJ at 32% — uranium near 40% limit
        positions = {
            "CCJ": Position(
                symbol="CCJ",
                shares=320,
                average_cost=100.0,
                current_price=100.0,
                market_value=32000.0,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
            ),
        }
        portfolio = PortfolioState(
            positions=positions,
            buying_power=68000.0,
            cash=68000.0,
            total_equity=100000.0,
            market_value=32000.0,
        )

        check = PortfolioRiskCheck(settings)
        context = RiskCheckContext(
            symbol="URNM",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 50.0},
            portfolio=portfolio,
            current_price=50.0,
        )

        result = check.check(context)
        assert result.passes is True
        assert any("near limit" in w.lower() for w in result.warnings)


class TestDrawdownCircuitBreaker:
    """Tests for drawdown circuit breaker check."""

    def test_drawdown_triggers_size_reduction(self, settings):
        """When drawdown exceeds threshold, should flag size reduction."""
        mock_pm = MagicMock()
        # Peak was 10000, current is 8800 → 12% drawdown (> 10% threshold)
        mock_pm.get_peak_equity.return_value = 10000.0

        portfolio = PortfolioState(
            buying_power=8800.0,
            cash=8800.0,
            total_equity=8800.0,
        )

        check = PortfolioRiskCheck(settings, portfolio_manager=mock_pm)
        context = RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0},
            portfolio=portfolio,
            current_price=100.0,
        )

        result = check.check(context)
        # Drawdown check never rejects — it only warns
        assert result.passes is True
        assert any("drawdown" in w.lower() for w in result.warnings)
        assert context.metadata.get("drawdown_size_reduction") == 0.50

    def test_no_drawdown_no_reduction(self, settings):
        """When current equity equals peak, no reduction applied."""
        mock_pm = MagicMock()
        mock_pm.get_peak_equity.return_value = 50000.0

        portfolio = PortfolioState(
            buying_power=50000.0,
            cash=50000.0,
            total_equity=50000.0,
        )

        check = PortfolioRiskCheck(settings, portfolio_manager=mock_pm)
        context = RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0},
            portfolio=portfolio,
            current_price=100.0,
        )

        result = check.check(context)
        assert result.passes is True
        assert context.metadata.get("drawdown_size_reduction") is None

    def test_small_drawdown_no_reduction(self, settings):
        """Drawdown below threshold should not trigger reduction."""
        mock_pm = MagicMock()
        # Peak 10000, current 9500 → 5% drawdown (< 10% threshold)
        mock_pm.get_peak_equity.return_value = 10000.0

        portfolio = PortfolioState(
            buying_power=9500.0,
            cash=9500.0,
            total_equity=9500.0,
        )

        check = PortfolioRiskCheck(settings, portfolio_manager=mock_pm)
        context = RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0},
            portfolio=portfolio,
            current_price=100.0,
        )

        result = check.check(context)
        assert result.passes is True
        assert context.metadata.get("drawdown_size_reduction") is None
        assert not any("drawdown active" in w.lower() for w in result.warnings)

    def test_no_portfolio_manager_skips_check(self, settings):
        """Without portfolio manager, uses current equity as peak (no drawdown)."""
        portfolio = PortfolioState(
            buying_power=50000.0,
            cash=50000.0,
            total_equity=50000.0,
        )

        check = PortfolioRiskCheck(settings, portfolio_manager=None)
        context = RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0},
            portfolio=portfolio,
            current_price=100.0,
        )

        result = check.check(context)
        assert result.passes is True
        assert context.metadata.get("drawdown_size_reduction") is None


class TestPositionSizingWithDrawdown:
    """Tests for position sizing with drawdown reduction applied."""

    def test_drawdown_reduces_position_size(self, settings):
        """Position sizing should be halved when drawdown reduction is set."""
        check = PositionSizingCheck(settings)

        portfolio = PortfolioState(
            buying_power=50000.0,
            cash=50000.0,
            total_equity=50000.0,
        )

        # Use high ATR so ATR-based sizing stays under the 20% position cap
        # ATR=10 → stop_distance=20, risk=1000, shares=50, dollar=$5000 (< $10000 cap)
        normal_context = RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0, "atr_14": 10.0},
            portfolio=portfolio,
            current_price=100.0,
        )
        normal_result = check.check(normal_context)
        assert normal_result.passes is True
        normal_shares = normal_result.recommended_shares

        # Now with drawdown reduction
        drawdown_context = RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0, "atr_14": 10.0},
            portfolio=portfolio,
            current_price=100.0,
            metadata={"drawdown_size_reduction": 0.50},
        )
        drawdown_result = check.check(drawdown_context)
        assert drawdown_result.passes is True
        assert drawdown_result.recommended_shares == int(normal_shares * 0.5)
        assert any("drawdown" in w.lower() for w in drawdown_result.warnings)
