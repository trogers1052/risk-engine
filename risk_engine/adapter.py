"""
Risk adapter for decision-engine integration.

Provides a simple interface for the decision-engine to call risk checks.
"""

import logging
from typing import Dict, List, Optional

from .checker import RiskChecker
from .config import RiskSettings, load_settings
from .data.market_data import MarketDataLoader
from .data.portfolio_state import PortfolioStateManager
from .data.rules_client import RulesClient
from .models.risk_result import RiskCheckResult
from .calculators.stop_loss import StopLossCalculator, StopLossCalculation
from .alerts.publisher import AlertPublisher, AlertType

logger = logging.getLogger(__name__)


class RiskAdapter:
    """
    Adapter for decision-engine integration.

    Provides a simple, single-method interface for risk checking
    that can be imported and used by the decision-engine.

    Usage in decision-engine:
        from risk_engine import RiskAdapter

        risk_adapter = RiskAdapter()
        risk_adapter.initialize()

        # In signal processing:
        result = risk_adapter.check_risk(
            symbol="AAPL",
            signal_type="BUY",
            confidence=0.75,
            indicators={"close": 150.0, "atr_14": 3.5, ...}
        )

        if result.passes:
            # Proceed with trade
            publish_decision(signal, risk_result=result)
        else:
            # Rejected by risk engine
            logger.info(f"Risk rejected: {result.reason}")
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        settings: Optional[RiskSettings] = None,
        kafka_brokers: Optional[List[str]] = None,
    ):
        """
        Initialize the risk adapter.

        Args:
            config_path: Optional path to risk_config.yaml
            settings: Optional pre-configured RiskSettings
            kafka_brokers: Optional Kafka brokers for alerts
        """
        self.config_path = config_path
        self._settings = settings
        self._kafka_brokers = kafka_brokers or ["localhost:19092"]
        self._checker: Optional[RiskChecker] = None
        self._portfolio_manager: Optional[PortfolioStateManager] = None
        self._market_data: Optional[MarketDataLoader] = None
        self._rules_client: Optional[RulesClient] = None
        self._stop_loss_calc: Optional[StopLossCalculator] = None
        self._alert_publisher: Optional[AlertPublisher] = None
        self._initialized = False

    @property
    def settings(self) -> RiskSettings:
        """Get or load settings."""
        if self._settings is None:
            self._settings = load_settings(self.config_path)
        return self._settings

    def initialize(self) -> bool:
        """
        Initialize connections and risk checker.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            logger.info("Initializing risk engine adapter...")

            # Initialize portfolio state manager (Redis)
            self._portfolio_manager = PortfolioStateManager(self.settings)
            if not self._portfolio_manager.connect():
                logger.warning(
                    "Failed to connect to Redis - portfolio data unavailable"
                )
                # Continue without portfolio data

            # Initialize market data loader (TimescaleDB)
            self._market_data = MarketDataLoader(self.settings)
            if not self._market_data.connect():
                logger.warning(
                    "Failed to connect to TimescaleDB - VaR/correlation unavailable"
                )
                # Continue without market data

            # Initialize rules client (Redis)
            self._rules_client = RulesClient(self.settings)
            if not self._rules_client.connect():
                logger.warning(
                    "Failed to connect to Redis for rules - using defaults"
                )
                # Continue with default rules

            # Initialize stop loss calculator
            if self._portfolio_manager and self._rules_client:
                self._stop_loss_calc = StopLossCalculator(
                    settings=self.settings,
                    portfolio_manager=self._portfolio_manager,
                    rules_client=self._rules_client,
                )
                logger.info("Stop loss calculator initialized")

            # Initialize alert publisher
            self._alert_publisher = AlertPublisher(
                brokers=self._kafka_brokers,
                topic="trading.risk-alerts",
            )
            if self._alert_publisher.connect():
                logger.info("Alert publisher connected to Kafka")
            else:
                logger.warning("Alert publisher failed to connect - alerts will be logged only")

            # Initialize risk checker
            self._checker = RiskChecker(
                settings=self.settings,
                portfolio_manager=self._portfolio_manager,
                market_data=self._market_data,
            )

            self._initialized = True
            logger.info("Risk engine adapter initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize risk adapter: {e}", exc_info=True)
            return False

    def check_risk(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        indicators: Dict[str, float],
        current_price: Optional[float] = None,
    ) -> RiskCheckResult:
        """
        Check risk for a potential trade.

        This is the main entry point for the decision-engine.

        Args:
            symbol: Stock symbol
            signal_type: "BUY", "SELL", or "WATCH"
            confidence: Signal confidence (0-1)
            indicators: Dict of current indicator values
            current_price: Optional current price (extracted from indicators if not provided)

        Returns:
            RiskCheckResult with decision and sizing recommendation
        """
        if not self._initialized:
            if not self.initialize():
                return RiskCheckResult.reject(
                    reason="Risk engine not initialized",
                    risk_score=1.0,
                    symbol=symbol,
                )

        # Only check BUY signals (SELL signals pass through)
        if signal_type != "BUY":
            return RiskCheckResult.approve(
                recommended_shares=0,
                max_shares=0,
                recommended_dollar_amount=0,
                risk_score=0.0,
                reason=f"{signal_type} signals not risk-checked",
                symbol=symbol,
            )

        return self._checker.check(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            indicators=indicators,
            current_price=current_price,
        )

    def check_buy(
        self,
        symbol: str,
        confidence: float,
        indicators: Dict[str, float],
    ) -> RiskCheckResult:
        """Convenience method for checking BUY signals."""
        return self.check_risk(
            symbol=symbol,
            signal_type="BUY",
            confidence=confidence,
            indicators=indicators,
        )

    def calculate_stop_loss(
        self,
        symbol: str,
        current_price: Optional[float] = None,
    ) -> Optional[StopLossCalculation]:
        """
        Calculate stop loss for an existing position.

        Args:
            symbol: Stock symbol
            current_price: Optional current price

        Returns:
            StopLossCalculation or None if no position
        """
        if not self._stop_loss_calc:
            logger.warning("Stop loss calculator not initialized")
            return None

        return self._stop_loss_calc.calculate(symbol, current_price)

    def calculate_stop_loss_for_entry(
        self,
        symbol: str,
        entry_price: float,
        shares: float,
        current_price: Optional[float] = None,
    ) -> StopLossCalculation:
        """
        Calculate stop loss for a new entry (before position exists).

        Args:
            symbol: Stock symbol
            entry_price: Entry/average price
            shares: Number of shares
            current_price: Current price (defaults to entry_price)

        Returns:
            StopLossCalculation
        """
        if not self._stop_loss_calc:
            logger.warning("Stop loss calculator not initialized")
            # Create a temporary calculator with default rules
            if self._portfolio_manager and self._rules_client:
                self._stop_loss_calc = StopLossCalculator(
                    settings=self.settings,
                    portfolio_manager=self._portfolio_manager,
                    rules_client=self._rules_client,
                )

        return self._stop_loss_calc.calculate_from_entry(
            symbol=symbol,
            entry_price=entry_price,
            shares=shares,
            current_price=current_price,
        )

    def send_stop_loss_alert(
        self,
        symbol: str,
        entry_price: float,
        shares: float,
        current_price: Optional[float] = None,
    ) -> bool:
        """
        Calculate stop loss and send alert for a new position.

        Call this after a BUY order is executed to remind user
        to set their stop loss.

        Args:
            symbol: Stock symbol
            entry_price: Entry/average price
            shares: Number of shares
            current_price: Current price (defaults to entry_price)

        Returns:
            True if alert sent successfully
        """
        calc = self.calculate_stop_loss_for_entry(
            symbol=symbol,
            entry_price=entry_price,
            shares=shares,
            current_price=current_price,
        )

        if not self._alert_publisher:
            logger.warning("Alert publisher not initialized")
            # Still log the stop loss info
            logger.info(f"Stop loss for {symbol}: {calc.format_alert_message()}")
            return False

        return self._alert_publisher.publish_stop_loss_alert(
            symbol=calc.symbol,
            shares=calc.shares,
            average_cost=calc.average_cost,
            stop_loss_price=calc.stop_loss_price,
            stop_loss_pct=calc.stop_loss_pct,
            current_price=calc.current_price,
            metadata={
                "profit_target_price": calc.profit_target_price,
                "profit_target_pct": calc.profit_target_pct,
            },
        )

    def get_all_stop_losses(
        self,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> List[StopLossCalculation]:
        """
        Get stop loss calculations for all positions.

        Args:
            current_prices: Optional dict of symbol -> current price

        Returns:
            List of StopLossCalculation
        """
        if not self._stop_loss_calc:
            return []

        return self._stop_loss_calc.calculate_all(current_prices)

    def check_triggered_stops(
        self,
        current_prices: Optional[Dict[str, float]] = None,
        send_alerts: bool = True,
    ) -> List[StopLossCalculation]:
        """
        Check for triggered stop losses and optionally send alerts.

        Args:
            current_prices: Optional dict of symbol -> current price
            send_alerts: Whether to send alerts for triggered stops

        Returns:
            List of triggered StopLossCalculations
        """
        if not self._stop_loss_calc:
            return []

        triggered = self._stop_loss_calc.get_triggered_stops(current_prices)

        if send_alerts and self._alert_publisher:
            for calc in triggered:
                self._alert_publisher.publish_stop_triggered_alert(
                    symbol=calc.symbol,
                    shares=calc.shares,
                    average_cost=calc.average_cost,
                    stop_loss_price=calc.stop_loss_price,
                    current_price=calc.current_price,
                    loss_amount=calc.potential_loss,
                )

        return triggered

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio state summary."""
        if not self._initialized or self._portfolio_manager is None:
            return {"error": "Not initialized"}

        state = self._portfolio_manager.get_portfolio_state()
        return state.to_dict()

    def get_position_weight(self, symbol: str) -> float:
        """Get current position weight for a symbol."""
        if not self._initialized or self._portfolio_manager is None:
            return 0.0

        return self._portfolio_manager.get_position_weight(symbol)

    def refresh_portfolio(self) -> None:
        """Force refresh of portfolio state."""
        if self._checker:
            self._checker.refresh_portfolio()

    def shutdown(self) -> None:
        """Clean up connections."""
        if self._portfolio_manager:
            self._portfolio_manager.close()
        if self._market_data:
            self._market_data.close()
        if self._rules_client:
            self._rules_client.close()
        if self._alert_publisher:
            self._alert_publisher.close()
        self._initialized = False
        logger.info("Risk adapter shut down")


# Singleton instance for easy import
_default_adapter: Optional[RiskAdapter] = None


def get_risk_adapter(
    config_path: Optional[str] = None,
    settings: Optional[RiskSettings] = None,
) -> RiskAdapter:
    """
    Get or create the default risk adapter instance.

    This provides a simple way to use the risk engine as a singleton.

    Usage:
        from risk_engine.adapter import get_risk_adapter

        adapter = get_risk_adapter()
        result = adapter.check_buy("AAPL", 0.75, indicators)
    """
    global _default_adapter

    if _default_adapter is None:
        _default_adapter = RiskAdapter(config_path, settings)

    return _default_adapter
