"""
Risk checker orchestrator.

Coordinates all risk checks and produces final risk assessment.
"""

import logging
from typing import Dict, List, Optional

from .checks.base import RiskCheck, RiskCheckContext
from .checks.position_sizing import PositionSizingCheck
from .checks.portfolio_risk import PortfolioRiskCheck
from .checks.trade_risk import TradeRiskCheck
from .checks.var_check import VaRCheck
from .config import RiskSettings
from .data.market_data import MarketDataLoader
from .data.portfolio_state import PortfolioStateManager
from .models.portfolio import PortfolioState
from .models.risk_result import RiskCheckResult, RiskLevel

logger = logging.getLogger(__name__)


class RiskChecker:
    """
    Orchestrates risk checks for trade decisions.

    Runs a pipeline of risk checks:
    1. Trade risk (volatility, trend, liquidity)
    2. Portfolio risk (concentration, correlation, exposure)
    3. VaR/CVaR check
    4. Position sizing

    Returns a consolidated RiskCheckResult.
    """

    def __init__(
        self,
        settings: RiskSettings,
        portfolio_manager: Optional[PortfolioStateManager] = None,
        market_data: Optional[MarketDataLoader] = None,
    ):
        self.settings = settings
        self.portfolio_manager = portfolio_manager
        self.market_data = market_data

        # Initialize checks
        self._checks: List[RiskCheck] = self._initialize_checks()

    def _initialize_checks(self) -> List[RiskCheck]:
        """Initialize the risk check pipeline."""
        checks = []

        # Trade-level risk (first line of defense)
        checks.append(TradeRiskCheck(self.settings, self.market_data))

        # Portfolio-level risk
        checks.append(PortfolioRiskCheck(self.settings, self.market_data))

        # VaR/CVaR check (if enabled)
        if self.settings.var.enabled:
            checks.append(VaRCheck(self.settings, self.market_data))

        # Position sizing (always last)
        checks.append(PositionSizingCheck(self.settings))

        return checks

    def check(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        indicators: Dict[str, float],
        portfolio: Optional[PortfolioState] = None,
        current_price: Optional[float] = None,
    ) -> RiskCheckResult:
        """
        Run all risk checks for a potential trade.

        Args:
            symbol: Stock symbol
            signal_type: "BUY", "SELL", or "WATCH"
            confidence: Signal confidence (0-1)
            indicators: Dict of indicator values
            portfolio: Optional portfolio state (loaded if not provided)
            current_price: Optional current price (taken from indicators if not provided)

        Returns:
            Consolidated RiskCheckResult
        """
        # Load portfolio if not provided
        if portfolio is None and self.portfolio_manager is not None:
            portfolio = self.portfolio_manager.get_portfolio_state()

        # Extract price from indicators if not provided
        if current_price is None:
            current_price = (
                indicators.get("close")
                or indicators.get("price")
                or indicators.get("last_price")
            )

        # Build context
        context = RiskCheckContext(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            indicators=indicators,
            portfolio=portfolio,
            current_price=current_price,
        )

        logger.debug(
            f"Running risk checks for {symbol} {signal_type} "
            f"(confidence={confidence:.2f}, price=${current_price})"
        )

        # Run checks
        all_warnings: List[str] = []
        all_metrics: Dict[str, float] = {}
        position_result: Optional[RiskCheckResult] = None

        for check in self._checks:
            if not check.can_check(context):
                logger.debug(f"Skipping {check.name}: cannot evaluate")
                all_warnings.append(f"{check.name} skipped: missing data")
                continue

            result = check.check(context)

            # Collect warnings and metrics
            all_warnings.extend(result.warnings)
            all_metrics.update(result.risk_metrics)

            if not result.passes:
                # Check failed - return rejection
                result.warnings = all_warnings
                result.risk_metrics = all_metrics
                logger.info(
                    f"Risk check {check.name} rejected {symbol}: {result.reason}"
                )
                return result

            # Track position sizing result
            if check.name == "position_sizing":
                position_result = result

            logger.debug(
                f"Check {check.name} passed for {symbol} "
                f"(risk_score={result.risk_score:.2f})"
            )

        # All checks passed - build final result
        if position_result is None:
            # No position sizing result - use defaults
            return RiskCheckResult.approve(
                recommended_shares=0,
                max_shares=0,
                recommended_dollar_amount=0,
                risk_score=0.5,
                reason="All risk checks passed (no sizing)",
                symbol=symbol,
                warnings=all_warnings,
                risk_metrics=all_metrics,
            )

        # Use position sizing result as base
        final_result = RiskCheckResult.approve(
            recommended_shares=position_result.recommended_shares,
            max_shares=position_result.max_shares,
            recommended_dollar_amount=position_result.recommended_dollar_amount,
            risk_score=self._calculate_aggregate_risk_score(all_metrics),
            reason="All risk checks passed",
            symbol=symbol,
            warnings=all_warnings,
            risk_metrics=all_metrics,
        )

        logger.info(
            f"Risk checks passed for {symbol}: "
            f"{final_result.recommended_shares} shares "
            f"(${final_result.recommended_dollar_amount:.2f}), "
            f"risk_score={final_result.risk_score:.2f}"
        )

        return final_result

    def check_buy(
        self,
        symbol: str,
        confidence: float,
        indicators: Dict[str, float],
    ) -> RiskCheckResult:
        """Convenience method for BUY signal checks."""
        return self.check(
            symbol=symbol,
            signal_type="BUY",
            confidence=confidence,
            indicators=indicators,
        )

    def check_sell(
        self,
        symbol: str,
        confidence: float,
        indicators: Dict[str, float],
    ) -> RiskCheckResult:
        """Convenience method for SELL signal checks."""
        return self.check(
            symbol=symbol,
            signal_type="SELL",
            confidence=confidence,
            indicators=indicators,
        )

    def _calculate_aggregate_risk_score(
        self,
        metrics: Dict[str, float],
    ) -> float:
        """Calculate aggregate risk score from all metrics."""
        scores = []

        # VaR contribution
        if "var_95" in metrics:
            max_var = self.settings.var.max_var_daily_pct
            scores.append(metrics["var_95"] / max_var)

        # CVaR contribution
        if "cvar_95" in metrics:
            max_cvar = self.settings.var.max_cvar_daily_pct
            scores.append(metrics["cvar_95"] / max_cvar)

        # Exposure contribution
        if "current_exposure" in metrics:
            max_exposure = self.settings.portfolio_risk.max_total_exposure_pct
            scores.append(metrics["current_exposure"] / max_exposure)

        # Concentration contribution
        if "current_weight" in metrics:
            max_conc = self.settings.portfolio_risk.max_concentration_pct
            scores.append(metrics["current_weight"] / max_conc)

        # Volatility contribution
        if "atr_pct" in metrics:
            max_atr = self.settings.trade_risk.max_atr_pct
            scores.append(metrics["atr_pct"] / max_atr)

        if not scores:
            return 0.5

        # Average of all risk contributions
        return min(1.0, sum(scores) / len(scores))

    def get_portfolio_state(self) -> Optional[PortfolioState]:
        """Get current portfolio state."""
        if self.portfolio_manager:
            return self.portfolio_manager.get_portfolio_state()
        return None

    def refresh_portfolio(self) -> None:
        """Force refresh of portfolio state."""
        if self.portfolio_manager:
            self.portfolio_manager.get_portfolio_state(force_refresh=True)
