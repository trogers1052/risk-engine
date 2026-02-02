"""
Risk adapter for decision-engine integration.

Provides a simple interface for the decision-engine to call risk checks.
"""

import logging
from typing import Dict, Optional

from .checker import RiskChecker
from .config import RiskSettings, load_settings
from .data.market_data import MarketDataLoader
from .data.portfolio_state import PortfolioStateManager
from .models.risk_result import RiskCheckResult

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
    ):
        """
        Initialize the risk adapter.

        Args:
            config_path: Optional path to risk_config.yaml
            settings: Optional pre-configured RiskSettings
        """
        self.config_path = config_path
        self._settings = settings
        self._checker: Optional[RiskChecker] = None
        self._portfolio_manager: Optional[PortfolioStateManager] = None
        self._market_data: Optional[MarketDataLoader] = None
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
