"""Tests for risk checks."""

import pytest
from unittest.mock import MagicMock
from risk_engine.checks.base import RiskCheckContext
from risk_engine.checks.position_sizing import PositionSizingCheck
from risk_engine.checks.portfolio_risk import PortfolioRiskCheck
from risk_engine.checks.trade_risk import TradeRiskCheck
from risk_engine.config import PortfolioRiskConfig, RiskSettings
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

    def test_warns_near_position_limit(self):
        """Should warn when one away from max positions."""
        # Use higher heat limit so heat check doesn't reject first
        from risk_engine.config import PortfolioRiskConfig
        test_settings = RiskSettings(
            portfolio_risk=PortfolioRiskConfig(max_portfolio_heat_pct=0.20),
        )

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

        check = PortfolioRiskCheck(test_settings)
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

    def test_drawdown_halt_blocks_entries(self, settings):
        """When drawdown exceeds halt threshold (15%), should reject."""
        mock_pm = MagicMock()
        # Peak was 10000, current is 8400 → 16% drawdown (> 15% halt)
        mock_pm.get_peak_equity.return_value = 10000.0

        portfolio = PortfolioState(
            buying_power=8400.0,
            cash=8400.0,
            total_equity=8400.0,
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
        assert result.passes is False
        assert "drawdown halt" in result.reason.lower()

    def test_drawdown_between_thresholds_reduces_not_halts(self, settings):
        """12% drawdown (between 10% and 15%) should reduce, not halt."""
        mock_pm = MagicMock()
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
        assert result.passes is True
        assert any("drawdown active" in w.lower() for w in result.warnings)
        assert context.metadata.get("drawdown_size_reduction") == 0.50

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


class TestPortfolioHeatCheck:
    """Tests for portfolio heat (total risk) check."""

    def _settings_with_heat(self, max_heat=0.08):
        """Create settings with heat limit."""
        from risk_engine.config import PortfolioRiskConfig, PositionSizingConfig
        return RiskSettings(
            position_sizing=PositionSizingConfig(risk_per_trade_pct=0.015),
            portfolio_risk=PortfolioRiskConfig(max_portfolio_heat_pct=max_heat),
        )

    def _portfolio_with_positions(self, symbols, total_equity=10000.0):
        """Create portfolio with given symbols as positions."""
        positions = {}
        per_pos_value = total_equity * 0.10  # 10% each
        for sym in symbols:
            positions[sym] = Position(
                symbol=sym,
                shares=10,
                average_cost=per_pos_value / 10,
                current_price=per_pos_value / 10,
                market_value=per_pos_value,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
            )
        market_value = per_pos_value * len(symbols)
        return PortfolioState(
            positions=positions,
            buying_power=total_equity - market_value,
            cash=total_equity - market_value,
            total_equity=total_equity,
            market_value=market_value,
        )

    def test_heat_within_limit(self):
        """3 positions at 1.5% = 4.5% + new 1.5% = 6% < 8% → PASS."""
        settings = self._settings_with_heat(0.08)
        portfolio = self._portfolio_with_positions(["AAPL", "GOOGL", "MSFT"])
        check = PortfolioRiskCheck(settings)

        context = RiskCheckContext(
            symbol="AMZN",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0},
            portfolio=portfolio,
            current_price=100.0,
        )

        result = check.check(context)
        assert result.passes is True

    def test_heat_exceeds_limit(self):
        """4 positions at 1.5% = 6% + new 1.5% = 7.5% > 7% limit → REJECT."""
        settings = self._settings_with_heat(0.07)
        portfolio = self._portfolio_with_positions(
            ["AAPL", "GOOGL", "MSFT", "AMZN"]
        )
        check = PortfolioRiskCheck(settings)

        context = RiskCheckContext(
            symbol="META",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0},
            portfolio=portfolio,
            current_price=100.0,
        )

        result = check.check(context)
        assert result.passes is False
        assert "heat" in result.reason.lower()

    def test_heat_with_reduced_risk_symbols(self):
        """MP at 0.75% risk counts less toward heat."""
        settings = self._settings_with_heat(0.08)
        settings = RiskSettings(
            position_sizing=settings.position_sizing,
            portfolio_risk=settings.portfolio_risk,
            symbol_overrides={"MP": {"risk_per_trade_pct": 0.0075}},
        )
        # 4 positions: 3 at 1.5% + MP at 0.75% = 5.25% + new 1.5% = 6.75%
        portfolio = self._portfolio_with_positions(
            ["AAPL", "GOOGL", "MSFT", "MP"]
        )
        check = PortfolioRiskCheck(settings)

        context = RiskCheckContext(
            symbol="AMZN",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0},
            portfolio=portfolio,
            current_price=100.0,
        )

        result = check.check(context)
        assert result.passes is True

    def test_heat_exactly_at_limit_passes(self):
        """Projected heat exactly at limit (not >) → PASS."""
        # 4 positions at 2% = 8%, adding 2% new = 10%
        # But if we set risk at 1.6% and have 4: 6.4% + 1.6% = 8.0% exactly
        from risk_engine.config import PortfolioRiskConfig, PositionSizingConfig
        settings = RiskSettings(
            position_sizing=PositionSizingConfig(risk_per_trade_pct=0.016),
            portfolio_risk=PortfolioRiskConfig(max_portfolio_heat_pct=0.08),
        )
        portfolio = self._portfolio_with_positions(
            ["AAPL", "GOOGL", "MSFT", "AMZN"]
        )
        check = PortfolioRiskCheck(settings)

        context = RiskCheckContext(
            symbol="META",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0},
            portfolio=portfolio,
            current_price=100.0,
        )

        result = check.check(context)
        # 0.08 is NOT > 0.08, so it passes
        assert result.passes is True

    def test_heat_empty_portfolio(self):
        """0 positions → PASS."""
        settings = self._settings_with_heat(0.08)
        portfolio = PortfolioState(
            buying_power=10000.0,
            cash=10000.0,
            total_equity=10000.0,
        )
        check = PortfolioRiskCheck(settings)

        context = RiskCheckContext(
            symbol="AAPL",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0},
            portfolio=portfolio,
            current_price=100.0,
        )

        result = check.check(context)
        assert result.passes is True

    def test_heat_warning_near_limit(self):
        """Projected heat near limit → PASS with warning."""
        # 4 positions at 1.5% = 6% + new 1.5% = 7.5% — within 1% of 8%
        settings = self._settings_with_heat(0.08)
        portfolio = self._portfolio_with_positions(
            ["AAPL", "GOOGL", "MSFT", "AMZN"]
        )
        check = PortfolioRiskCheck(settings)

        context = RiskCheckContext(
            symbol="META",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0},
            portfolio=portfolio,
            current_price=100.0,
        )

        result = check.check(context)
        assert result.passes is True
        assert any("heat near limit" in w.lower() for w in result.warnings)


class TestSectorPositionCount:
    """Tests for sector position count limit."""

    def _settings_with_sectors(self, max_sector_positions=2):
        """Create settings with sector groups and position count limit."""
        from risk_engine.config import PortfolioRiskConfig
        return RiskSettings(
            sector_groups={
                "uranium": ["CCJ", "URNM", "UUUU"],
                "precious_metals": ["WPM", "SLV", "IAUM", "PPLT"],
                "industrial": ["CAT", "ETN"],
            },
            portfolio_risk=PortfolioRiskConfig(
                max_sector_positions=max_sector_positions,
                max_portfolio_heat_pct=1.0,  # disable heat check for these tests
            ),
        )

    def test_sector_count_within_limit(self):
        """1 uranium position + new CCJ = 2 → PASS."""
        settings = self._settings_with_sectors()
        positions = {
            "URNM": Position(
                symbol="URNM",
                shares=100,
                average_cost=30.0,
                current_price=30.0,
                market_value=3000.0,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
            ),
        }
        portfolio = PortfolioState(
            positions=positions,
            buying_power=7000.0,
            cash=7000.0,
            total_equity=10000.0,
            market_value=3000.0,
        )

        check = PortfolioRiskCheck(settings)
        context = RiskCheckContext(
            symbol="CCJ",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 50.0},
            portfolio=portfolio,
            current_price=50.0,
        )

        result = check.check(context)
        assert result.passes is True

    def test_sector_count_at_limit(self):
        """2 uranium positions + new UUUU = 3 → REJECT."""
        settings = self._settings_with_sectors()
        positions = {
            "CCJ": Position(
                symbol="CCJ",
                shares=50,
                average_cost=50.0,
                current_price=50.0,
                market_value=2500.0,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
            ),
            "URNM": Position(
                symbol="URNM",
                shares=100,
                average_cost=30.0,
                current_price=30.0,
                market_value=3000.0,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
            ),
        }
        portfolio = PortfolioState(
            positions=positions,
            buying_power=50000.0,
            cash=50000.0,
            total_equity=55500.0,
            market_value=5500.0,
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
        assert "position limit" in result.reason.lower()
        assert "uranium" in result.reason.lower()

    def test_sector_count_different_sector(self):
        """2 uranium positions + new CAT (industrial) → PASS."""
        settings = self._settings_with_sectors()
        positions = {
            "CCJ": Position(
                symbol="CCJ",
                shares=50,
                average_cost=50.0,
                current_price=50.0,
                market_value=2500.0,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
            ),
            "URNM": Position(
                symbol="URNM",
                shares=100,
                average_cost=30.0,
                current_price=30.0,
                market_value=3000.0,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
            ),
        }
        portfolio = PortfolioState(
            positions=positions,
            buying_power=50000.0,
            cash=50000.0,
            total_equity=55500.0,
            market_value=5500.0,
        )

        check = PortfolioRiskCheck(settings)
        context = RiskCheckContext(
            symbol="CAT",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 300.0},
            portfolio=portfolio,
            current_price=300.0,
        )

        result = check.check(context)
        assert result.passes is True

    def test_sector_count_unmapped_symbol(self):
        """Symbol not in any sector → PASS with warning."""
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

    def test_sector_count_scale_in(self):
        """Already hold CCJ, buying more CCJ → PASS (not a new position)."""
        settings = self._settings_with_sectors()
        positions = {
            "CCJ": Position(
                symbol="CCJ",
                shares=50,
                average_cost=50.0,
                current_price=50.0,
                market_value=2500.0,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
            ),
            "URNM": Position(
                symbol="URNM",
                shares=100,
                average_cost=30.0,
                current_price=30.0,
                market_value=3000.0,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
            ),
        }
        portfolio = PortfolioState(
            positions=positions,
            buying_power=50000.0,
            cash=50000.0,
            total_equity=55500.0,
            market_value=5500.0,
        )

        check = PortfolioRiskCheck(settings)
        context = RiskCheckContext(
            symbol="CCJ",  # scale-in — already held
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 50.0},
            portfolio=portfolio,
            current_price=50.0,
        )

        result = check.check(context)
        assert result.passes is True

    def test_existing_pct_check_still_works(self):
        """% exposure check still rejects when appropriate."""
        settings = self._settings_with_sectors(max_sector_positions=5)
        # 1 position at 42% of equity — exceeds 40% sector limit
        positions = {
            "CCJ": Position(
                symbol="CCJ",
                shares=420,
                average_cost=100.0,
                current_price=100.0,
                market_value=42000.0,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
            ),
        }
        portfolio = PortfolioState(
            positions=positions,
            buying_power=58000.0,
            cash=58000.0,
            total_equity=100000.0,
            market_value=42000.0,
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
        assert result.passes is False
        assert "exposure" in result.reason.lower()
        assert "uranium" in result.reason.lower()


class TestCapitalTemperature:
    """Tests for capital temperature gauge."""

    @staticmethod
    def _make_check(vix=None, hy=None, enabled=True, **config_overrides):
        """Create a PortfolioRiskCheck with mocked macro data."""
        from risk_engine.config import CapitalTemperatureConfig

        temp_config = CapitalTemperatureConfig(enabled=enabled, **config_overrides)
        settings = RiskSettings(
            capital_temperature=temp_config,
            portfolio_risk=PortfolioRiskConfig(
                max_portfolio_heat_pct=0.20,  # high to avoid heat rejection
            ),
        )

        mock_pm = MagicMock()
        macro = {}
        if vix is not None:
            macro["vix"] = vix
        if hy is not None:
            macro["hy_spread"] = hy
        mock_pm.get_macro_signals.return_value = macro
        mock_pm.get_peak_equity.return_value = 10000.0

        check = PortfolioRiskCheck(settings, portfolio_manager=mock_pm)
        return check

    @staticmethod
    def _make_context(portfolio=None):
        if portfolio is None:
            portfolio = PortfolioState(
                buying_power=8000.0,
                cash=8000.0,
                total_equity=10000.0,
                market_value=2000.0,
            )
        return RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 50.0},
            portfolio=portfolio,
            current_price=50.0,
        )

    def test_temperature_calm_market(self):
        """VIX=12, HY=3.0 → temperature=0, no reduction."""
        check = self._make_check(vix=12.0, hy=3.0)
        context = self._make_context()

        result = check.check(context)
        assert result.passes is True
        assert "capital_temperature_reduction" not in context.metadata
        assert result.risk_metrics.get("capital_temperature", 0) == 0.0

    def test_temperature_elevated_vix(self):
        """VIX=30, HY=3.0 → reduction ~11%."""
        check = self._make_check(vix=30.0, hy=3.0)
        context = self._make_context()

        result = check.check(context)
        assert result.passes is True

        # vix_score = (30-15)/25 = 0.6, hy_score = 0 (below floor)
        # temperature = 0.6*0.6 + 0.4*0 = 0.36
        # reduction = 0.36 * 0.30 = 0.108
        reduction = context.metadata.get("capital_temperature_reduction", 0)
        assert 0.10 <= reduction <= 0.12
        assert any("capital temperature" in w.lower() for w in result.warnings)

    def test_temperature_crisis(self):
        """VIX=40, HY=8.0 → max reduction (30%)."""
        check = self._make_check(vix=40.0, hy=8.0)
        context = self._make_context()

        result = check.check(context)
        assert result.passes is True

        # vix_score = 1.0, hy_score = 1.0
        # temperature = 0.6*1.0 + 0.4*1.0 = 1.0
        # reduction = 1.0 * 0.30 = 0.30
        reduction = context.metadata.get("capital_temperature_reduction", 0)
        assert abs(reduction - 0.30) < 0.01

    def test_temperature_no_macro_data(self):
        """No macro data → skipped with warning."""
        check = self._make_check()
        check.portfolio_manager.get_macro_signals.return_value = {}
        context = self._make_context()

        result = check.check(context)
        assert result.passes is True
        assert "capital_temperature_reduction" not in context.metadata
        assert any("no macro data" in w.lower() for w in result.warnings)

    def test_temperature_disabled(self):
        """Disabled config → no temperature check."""
        check = self._make_check(vix=40.0, hy=8.0, enabled=False)
        context = self._make_context()

        result = check.check(context)
        assert result.passes is True
        assert "capital_temperature_reduction" not in context.metadata

    def test_temperature_vix_only(self):
        """Only VIX available, no HY spread → uses VIX only."""
        check = self._make_check(vix=30.0)
        context = self._make_context()

        result = check.check(context)
        assert result.passes is True

        # vix_score = 0.6, hy_score = 0 (not available)
        # temperature = 0.6*0.6 = 0.36
        # reduction = 0.36 * 0.30 = 0.108
        reduction = context.metadata.get("capital_temperature_reduction", 0)
        assert 0.10 <= reduction <= 0.12


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

    def test_temperature_reduces_position_size(self, settings):
        """Position sizing should be reduced when temperature reduction is set."""
        check = PositionSizingCheck(settings)

        portfolio = PortfolioState(
            buying_power=50000.0,
            cash=50000.0,
            total_equity=50000.0,
        )

        # Normal sizing first
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

        # Now with 20% temperature reduction
        temp_context = RiskCheckContext(
            symbol="MSFT",
            signal_type="BUY",
            confidence=0.8,
            indicators={"close": 100.0, "atr_14": 10.0},
            portfolio=portfolio,
            current_price=100.0,
            metadata={"capital_temperature_reduction": 0.20},
        )
        temp_result = check.check(temp_context)
        assert temp_result.passes is True
        assert temp_result.recommended_shares == int(normal_shares * 0.8)
        assert any("capital temperature" in w.lower() for w in temp_result.warnings)
