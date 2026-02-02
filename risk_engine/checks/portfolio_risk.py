"""
Portfolio-level risk checks.

Evaluates concentration, correlation, and total exposure limits.
"""

import logging
from typing import List, Optional

from ..config import RiskSettings
from ..data.market_data import MarketDataLoader
from ..models.risk_result import RiskCheckResult
from .base import RiskCheck, RiskCheckContext

logger = logging.getLogger(__name__)


class PortfolioRiskCheck(RiskCheck):
    """
    Portfolio risk check for concentration and correlation.

    Checks:
    - Max concentration per symbol (default 20%)
    - Max correlation with existing positions (default 80%)
    - Max total exposure / min cash buffer (default 95%/5%)
    """

    def __init__(
        self,
        settings: RiskSettings,
        market_data: Optional[MarketDataLoader] = None,
    ):
        super().__init__(settings)
        self.market_data = market_data

    @property
    def name(self) -> str:
        return "portfolio_risk"

    @property
    def description(self) -> str:
        return "Check portfolio concentration, correlation, and exposure limits"

    def can_check(self, context: RiskCheckContext) -> bool:
        """Need portfolio state."""
        return context.portfolio is not None

    def check(self, context: RiskCheckContext) -> RiskCheckResult:
        """Evaluate portfolio-level risk."""
        portfolio = context.portfolio
        symbol = context.symbol
        price = context.price

        if portfolio is None:
            return self._reject(
                "No portfolio data",
                risk_score=1.0,
                context=context,
            )

        warnings: List[str] = []
        risk_metrics = {}

        # Check 1: Total exposure limit
        exposure_result = self._check_exposure(context, warnings, risk_metrics)
        if exposure_result is not None:
            return exposure_result

        # Check 2: Concentration limit
        concentration_result = self._check_concentration(
            context, warnings, risk_metrics
        )
        if concentration_result is not None:
            return concentration_result

        # Check 3: Correlation with existing positions
        correlation_result = self._check_correlation(
            context, warnings, risk_metrics
        )
        if correlation_result is not None:
            return correlation_result

        # All checks passed - calculate aggregate risk score
        risk_score = self._calculate_aggregate_risk(risk_metrics)

        # Determine position sizing based on checks
        max_position_pct = self.settings.get_max_position_pct(symbol)
        available_capacity = 1.0 - portfolio.get_total_exposure()
        effective_max_pct = min(max_position_pct, available_capacity)

        if price and price > 0:
            max_dollar = portfolio.total_equity * effective_max_pct
            max_shares = int(max_dollar / price)
        else:
            max_dollar = 0
            max_shares = 0

        return self._approve(
            recommended_shares=max_shares,
            max_shares=max_shares,
            recommended_dollar_amount=max_dollar,
            risk_score=risk_score,
            context=context,
            reason="Portfolio risk checks passed",
            warnings=warnings,
            risk_metrics=risk_metrics,
        )

    def _check_exposure(
        self,
        context: RiskCheckContext,
        warnings: List[str],
        risk_metrics: dict,
    ) -> Optional[RiskCheckResult]:
        """Check total portfolio exposure limit."""
        portfolio = context.portfolio
        max_exposure = self.settings.portfolio_risk.max_total_exposure_pct
        min_cash = self.settings.portfolio_risk.min_cash_pct

        current_exposure = portfolio.get_total_exposure()
        risk_metrics["current_exposure"] = current_exposure
        risk_metrics["max_exposure"] = max_exposure

        if current_exposure >= max_exposure:
            return self._reject(
                f"Total exposure {current_exposure:.1%} exceeds "
                f"limit {max_exposure:.1%}",
                risk_score=0.9,
                context=context,
                risk_metrics=risk_metrics,
            )

        # Check cash buffer
        cash_pct = portfolio.cash_pct
        risk_metrics["cash_pct"] = cash_pct

        if cash_pct < min_cash:
            warnings.append(
                f"Cash buffer low: {cash_pct:.1%} (min {min_cash:.1%})"
            )

        return None

    def _check_concentration(
        self,
        context: RiskCheckContext,
        warnings: List[str],
        risk_metrics: dict,
    ) -> Optional[RiskCheckResult]:
        """Check position concentration limit."""
        portfolio = context.portfolio
        symbol = context.symbol
        max_concentration = self.settings.get_max_position_pct(symbol)

        # Check existing position weight
        current_weight = portfolio.get_position_weight(symbol)
        risk_metrics["current_weight"] = current_weight
        risk_metrics["max_concentration"] = max_concentration

        if current_weight >= max_concentration:
            return self._reject(
                f"Position {symbol} already at max concentration "
                f"({current_weight:.1%} >= {max_concentration:.1%})",
                risk_score=0.85,
                context=context,
                risk_metrics=risk_metrics,
            )

        # Check if adding would exceed
        remaining_capacity = max_concentration - current_weight
        risk_metrics["remaining_capacity"] = remaining_capacity

        if remaining_capacity < 0.01:  # Less than 1% capacity
            warnings.append(
                f"Limited capacity for {symbol}: {remaining_capacity:.1%} remaining"
            )

        return None

    def _check_correlation(
        self,
        context: RiskCheckContext,
        warnings: List[str],
        risk_metrics: dict,
    ) -> Optional[RiskCheckResult]:
        """Check correlation with existing positions."""
        if self.market_data is None:
            # Skip correlation check if no market data
            warnings.append("Correlation check skipped: no market data")
            return None

        portfolio = context.portfolio
        symbol = context.symbol
        max_correlation = self.settings.portfolio_risk.max_correlation

        # Get existing position symbols
        existing_symbols = [
            s for s in portfolio.symbols if s != symbol
        ]

        if not existing_symbols:
            # No existing positions to correlate with
            return None

        # Calculate correlations
        corr_symbol, max_corr = self.market_data.get_max_correlation_with_portfolio(
            symbol, existing_symbols, lookback_days=60
        )

        if corr_symbol is None:
            warnings.append(
                "Could not calculate correlations - insufficient data"
            )
            return None

        risk_metrics["max_correlation"] = max_corr
        risk_metrics["most_correlated_with"] = corr_symbol

        if abs(max_corr) > max_correlation:
            return self._reject(
                f"High correlation with {corr_symbol}: {max_corr:.2f} "
                f"(limit: {max_correlation})",
                risk_score=0.75,
                context=context,
                risk_metrics=risk_metrics,
            )

        if abs(max_corr) > max_correlation * 0.8:
            warnings.append(
                f"Moderate correlation with {corr_symbol}: {max_corr:.2f}"
            )

        return None

    def _calculate_aggregate_risk(self, risk_metrics: dict) -> float:
        """Calculate aggregate portfolio risk score."""
        scores = []

        # Exposure risk
        exposure = risk_metrics.get("current_exposure", 0)
        max_exposure = risk_metrics.get("max_exposure", 0.95)
        if max_exposure > 0:
            scores.append(exposure / max_exposure)

        # Concentration risk
        weight = risk_metrics.get("current_weight", 0)
        max_conc = risk_metrics.get("max_concentration", 0.20)
        if max_conc > 0:
            scores.append(weight / max_conc)

        # Correlation risk
        corr = abs(risk_metrics.get("max_correlation", 0))
        scores.append(corr)

        if not scores:
            return 0.5

        return sum(scores) / len(scores)
