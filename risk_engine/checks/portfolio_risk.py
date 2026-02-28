"""
Portfolio-level risk checks.

Evaluates concentration, correlation, exposure, sector limits,
position count, and drawdown circuit breaker.
"""

import logging
from typing import List, Optional

from ..config import RiskSettings
from ..data.market_data import MarketDataLoader
from ..data.portfolio_state import PortfolioStateManager
from ..models.risk_result import RiskCheckResult
from .base import RiskCheck, RiskCheckContext

logger = logging.getLogger(__name__)


class PortfolioRiskCheck(RiskCheck):
    """
    Portfolio risk check for concentration, correlation, sector exposure,
    position count, and drawdown circuit breaker.

    Checks (in order):
    1. Max open positions (default 5)
    2. Max total exposure / min cash buffer (default 95%/5%)
    3. Max concentration per symbol (default 20%)
    4. Max sector exposure (default 40%)
    5. Max correlation with existing positions (default 80%)
    6. Drawdown circuit breaker (warns + reduces sizing, default 10% threshold)
    """

    def __init__(
        self,
        settings: RiskSettings,
        market_data: Optional[MarketDataLoader] = None,
        portfolio_manager: Optional[PortfolioStateManager] = None,
    ):
        super().__init__(settings)
        self.market_data = market_data
        self.portfolio_manager = portfolio_manager

    @property
    def name(self) -> str:
        return "portfolio_risk"

    @property
    def description(self) -> str:
        return "Check portfolio concentration, correlation, sector, and exposure limits"

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

        # Check 1: Max open positions
        pos_result = self._check_open_positions(context, warnings, risk_metrics)
        if pos_result is not None:
            return pos_result

        # Check 2: Total exposure limit
        exposure_result = self._check_exposure(context, warnings, risk_metrics)
        if exposure_result is not None:
            return exposure_result

        # Check 3: Concentration limit
        concentration_result = self._check_concentration(
            context, warnings, risk_metrics
        )
        if concentration_result is not None:
            return concentration_result

        # Check 4: Sector concentration
        sector_result = self._check_sector_concentration(
            context, warnings, risk_metrics
        )
        if sector_result is not None:
            return sector_result

        # Check 5: Correlation with existing positions
        correlation_result = self._check_correlation(
            context, warnings, risk_metrics
        )
        if correlation_result is not None:
            return correlation_result

        # Check 6: Drawdown circuit breaker (warns, doesn't reject)
        self._check_drawdown_circuit_breaker(context, warnings, risk_metrics)

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

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_open_positions(
        self,
        context: RiskCheckContext,
        warnings: List[str],
        risk_metrics: dict,
    ) -> Optional[RiskCheckResult]:
        """Check max concurrent open positions."""
        max_positions = self.settings.portfolio_risk.max_open_positions
        current_count = context.portfolio.position_count
        risk_metrics["open_positions"] = current_count
        risk_metrics["max_positions"] = max_positions

        if current_count >= max_positions:
            return self._reject(
                f"Max positions reached: {current_count}/{max_positions}",
                risk_score=0.85,
                context=context,
                risk_metrics=risk_metrics,
            )

        if current_count >= max_positions - 1:
            warnings.append(
                f"Near position limit: {current_count}/{max_positions}"
            )

        return None

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

    def _check_sector_concentration(
        self,
        context: RiskCheckContext,
        warnings: List[str],
        risk_metrics: dict,
    ) -> Optional[RiskCheckResult]:
        """Check sector exposure limit.

        Aggregates position weights for all symbols in the same sector
        as the candidate symbol. Rejects if sector exposure is at/above
        the configured limit.
        """
        symbol = context.symbol
        sector = self.settings.get_sector_for_symbol(symbol)

        if sector is None:
            # Symbol not in any sector group — can't check, pass with warning
            warnings.append(f"No sector mapping for {symbol}")
            return None

        max_sector_pct = self.settings.portfolio_risk.max_sector_exposure_pct
        portfolio = context.portfolio

        # Sum weights of existing positions in same sector
        sector_symbols = self.settings.sector_groups.get(sector, [])
        sector_weight = sum(
            portfolio.get_position_weight(s) for s in sector_symbols
        )

        risk_metrics["sector"] = sector
        risk_metrics["sector_exposure"] = sector_weight
        risk_metrics["max_sector_exposure"] = max_sector_pct

        held_in_sector = [
            s for s in sector_symbols if portfolio.has_position(s)
        ]

        if sector_weight >= max_sector_pct:
            return self._reject(
                f"Sector '{sector}' exposure {sector_weight:.1%} "
                f"at/exceeds limit {max_sector_pct:.1%} "
                f"(positions: {held_in_sector})",
                risk_score=0.80,
                context=context,
                risk_metrics=risk_metrics,
            )

        remaining = max_sector_pct - sector_weight
        if remaining < 0.10:
            warnings.append(
                f"Sector '{sector}' near limit: {remaining:.1%} remaining "
                f"(positions: {held_in_sector})"
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

    def _check_drawdown_circuit_breaker(
        self,
        context: RiskCheckContext,
        warnings: List[str],
        risk_metrics: dict,
    ) -> None:
        """Check drawdown from peak equity and flag for position size reduction.

        This check never rejects — it sets a reduction factor in
        context.metadata["drawdown_size_reduction"] which the position
        sizing check applies downstream.
        """
        portfolio = context.portfolio
        threshold = self.settings.portfolio_risk.drawdown_threshold_pct
        current_equity = portfolio.total_equity

        if current_equity <= 0:
            return

        # Get peak equity (updates Redis if new high)
        if self.portfolio_manager is not None:
            peak_equity = self.portfolio_manager.get_peak_equity(current_equity)
        else:
            peak_equity = current_equity

        if peak_equity <= 0:
            return

        drawdown = (peak_equity - current_equity) / peak_equity
        risk_metrics["peak_equity"] = peak_equity
        risk_metrics["drawdown_pct"] = drawdown
        risk_metrics["drawdown_threshold"] = threshold

        if drawdown >= threshold:
            reduction = self.settings.portfolio_risk.drawdown_size_reduction
            risk_metrics["size_reduction"] = reduction
            warnings.append(
                f"DRAWDOWN ACTIVE: {drawdown:.1%} from peak ${peak_equity:.2f}. "
                f"Position sizes reduced by {reduction:.0%}"
            )
            # Flag for position sizing to apply reduction
            context.metadata["drawdown_size_reduction"] = reduction
        elif drawdown > 0:
            risk_metrics["size_reduction"] = 0.0

    # ------------------------------------------------------------------
    # Aggregate scoring
    # ------------------------------------------------------------------

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

        # Sector risk
        sector_exp = risk_metrics.get("sector_exposure", 0)
        max_sector = risk_metrics.get("max_sector_exposure", 0.40)
        if max_sector > 0 and sector_exp > 0:
            scores.append(sector_exp / max_sector)

        # Correlation risk
        corr = abs(risk_metrics.get("max_correlation", 0))
        if isinstance(corr, (int, float)):
            scores.append(corr)

        # Drawdown risk
        drawdown = risk_metrics.get("drawdown_pct", 0)
        if drawdown > 0:
            threshold = risk_metrics.get("drawdown_threshold", 0.10)
            scores.append(min(1.0, drawdown / threshold))

        if not scores:
            return 0.5

        return sum(scores) / len(scores)
