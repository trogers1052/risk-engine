"""
Position sizing risk check.

Calculates recommended position size using ATR-based or Kelly methods.
"""

import logging
from typing import List, Optional

from ..config import RiskSettings
from ..models.risk_result import RiskCheckResult
from .base import RiskCheck, RiskCheckContext

logger = logging.getLogger(__name__)


class PositionSizingCheck(RiskCheck):
    """
    Position sizing check using ATR or Kelly criterion.

    ATR-based: Position = (Portfolio * RiskPct) / (ATR * Multiplier)
    Kelly: Adjusted Kelly = (mean/variance) * kelly_fraction

    Also enforces maximum position size limits.
    """

    @property
    def name(self) -> str:
        return "position_sizing"

    @property
    def description(self) -> str:
        return "Calculate position size based on ATR or Kelly criterion"

    def can_check(self, context: RiskCheckContext) -> bool:
        """Need price and either ATR or historical data for Kelly."""
        if context.price is None:
            return False
        if self.settings.position_sizing.method == "atr":
            return context.atr is not None
        return True

    def check(self, context: RiskCheckContext) -> RiskCheckResult:
        """Calculate recommended position size."""
        price = context.price
        if price is None or price <= 0:
            return self._reject(
                "Invalid price",
                risk_score=1.0,
                context=context,
            )

        portfolio = context.portfolio
        if portfolio is None or portfolio.total_equity <= 0:
            return self._reject(
                "No portfolio data available",
                risk_score=1.0,
                context=context,
            )

        method = self.settings.position_sizing.method
        warnings: List[str] = []

        # Calculate position size based on method
        if method == "kelly":
            result = self._kelly_sizing(context, warnings)
        else:  # Default to ATR
            result = self._atr_sizing(context, warnings)

        if result is None:
            return self._reject(
                f"Could not calculate {method} position size",
                risk_score=0.8,
                context=context,
                warnings=warnings,
            )

        recommended_shares, dollar_amount, risk_score = result

        # Apply maximum position limit
        max_position_pct = self.settings.get_max_position_pct(context.symbol)
        max_dollar_amount = portfolio.total_equity * max_position_pct
        max_shares = int(max_dollar_amount / price)

        if dollar_amount > max_dollar_amount:
            warnings.append(
                f"Position capped from ${dollar_amount:.2f} to "
                f"${max_dollar_amount:.2f} (max {max_position_pct*100:.0f}%)"
            )
            recommended_shares = min(recommended_shares, max_shares)
            dollar_amount = recommended_shares * price

        # Check buying power
        if portfolio.buying_power < dollar_amount:
            if portfolio.buying_power > price:
                affordable_shares = int(portfolio.buying_power / price)
                warnings.append(
                    f"Limited by buying power: {affordable_shares} shares "
                    f"(${portfolio.buying_power:.2f} available)"
                )
                recommended_shares = affordable_shares
                dollar_amount = recommended_shares * price
            else:
                return self._reject(
                    f"Insufficient buying power: ${portfolio.buying_power:.2f}",
                    risk_score=0.9,
                    context=context,
                    warnings=warnings,
                )

        if recommended_shares <= 0:
            return self._reject(
                "Position size too small",
                risk_score=0.7,
                context=context,
                warnings=warnings,
            )

        return self._approve(
            recommended_shares=recommended_shares,
            max_shares=max_shares,
            recommended_dollar_amount=dollar_amount,
            risk_score=risk_score,
            context=context,
            reason=f"{method.upper()} sizing: {recommended_shares} shares",
            warnings=warnings,
            risk_metrics={
                "position_pct": dollar_amount / portfolio.total_equity,
                "max_position_pct": max_position_pct,
            },
        )

    def _atr_sizing(
        self,
        context: RiskCheckContext,
        warnings: List[str],
    ) -> Optional[tuple]:
        """
        Calculate position size using ATR-based method.

        Formula: Position = (Portfolio * RiskPct) / (ATR * Multiplier)
        """
        atr = context.atr
        price = context.price
        portfolio = context.portfolio

        if atr is None or atr <= 0:
            warnings.append("ATR not available, cannot calculate position size")
            return None

        risk_pct = self.settings.position_sizing.risk_per_trade_pct
        multiplier = self.settings.position_sizing.atr_multiplier

        # Risk amount = portfolio * risk percentage
        risk_amount = portfolio.total_equity * risk_pct

        # Stop distance = ATR * multiplier
        stop_distance = atr * multiplier

        # Shares = risk amount / stop distance
        shares = int(risk_amount / stop_distance)
        dollar_amount = shares * price

        # Calculate risk score (lower ATR = lower risk)
        atr_pct = atr / price if price > 0 else 0
        risk_score = min(1.0, atr_pct / 0.10)  # 10% ATR = max risk

        logger.debug(
            f"ATR sizing {context.symbol}: "
            f"ATR={atr:.2f}, stop={stop_distance:.2f}, "
            f"shares={shares}, amount=${dollar_amount:.2f}"
        )

        return shares, dollar_amount, risk_score

    def _kelly_sizing(
        self,
        context: RiskCheckContext,
        warnings: List[str],
    ) -> Optional[tuple]:
        """
        Calculate position size using Kelly criterion.

        Formula: f* = (mean - rf) / variance
        Adjusted: position = f* * kelly_fraction
        """
        # Kelly requires historical returns - we'd need to fetch them
        # For now, use a simplified version based on confidence and ATR
        price = context.price
        portfolio = context.portfolio
        confidence = context.confidence

        # Use fractional Kelly based on signal confidence
        kelly_fraction = self.settings.position_sizing.kelly_fraction

        # Base allocation from confidence
        # Higher confidence = larger position
        base_pct = confidence * kelly_fraction

        # Cap at max position
        max_pct = self.settings.get_max_position_pct(context.symbol)
        position_pct = min(base_pct, max_pct)

        dollar_amount = portfolio.total_equity * position_pct
        shares = int(dollar_amount / price)

        # Risk score based on confidence (inverse - high confidence = low risk)
        risk_score = 1.0 - (confidence * 0.5)

        warnings.append(
            f"Kelly sizing based on confidence ({confidence:.2f}) "
            f"with fraction {kelly_fraction}"
        )

        logger.debug(
            f"Kelly sizing {context.symbol}: "
            f"confidence={confidence:.2f}, fraction={kelly_fraction}, "
            f"pct={position_pct:.2%}, shares={shares}"
        )

        return shares, dollar_amount, risk_score
