"""
Trade-level risk checks.

Evaluates volatility, trend alignment, and liquidity.
"""

import logging
from typing import List, Optional

from ..config import RiskSettings
from ..data.market_data import MarketDataLoader
from ..models.risk_result import RiskCheckResult
from .base import RiskCheck, RiskCheckContext

logger = logging.getLogger(__name__)


class TradeRiskCheck(RiskCheck):
    """
    Trade-level risk check for individual trade quality.

    Checks:
    - Volatility: Reject if ATR > 5% of price
    - Trend: BUY only above 200 SMA
    - Liquidity: Min 50% of average volume
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
        return "trade_risk"

    @property
    def description(self) -> str:
        return "Check volatility, trend, and liquidity conditions"

    def can_check(self, context: RiskCheckContext) -> bool:
        """Need price for basic checks."""
        return context.price is not None and context.price > 0

    def check(self, context: RiskCheckContext) -> RiskCheckResult:
        """Evaluate trade-level risk."""
        warnings: List[str] = []
        risk_metrics = {}

        # Check 1: Volatility
        volatility_result = self._check_volatility(
            context, warnings, risk_metrics
        )
        if volatility_result is not None:
            return volatility_result

        # Check 2: Trend alignment (for BUY signals)
        if context.signal_type == "BUY":
            trend_result = self._check_trend(context, warnings, risk_metrics)
            if trend_result is not None:
                return trend_result

        # Check 3: Liquidity
        liquidity_result = self._check_liquidity(
            context, warnings, risk_metrics
        )
        if liquidity_result is not None:
            return liquidity_result

        # All checks passed
        risk_score = self._calculate_trade_risk(risk_metrics)

        return self._approve(
            recommended_shares=0,  # Trade risk doesn't size positions
            max_shares=0,
            recommended_dollar_amount=0,
            risk_score=risk_score,
            context=context,
            reason="Trade conditions acceptable",
            warnings=warnings,
            risk_metrics=risk_metrics,
        )

    def _check_volatility(
        self,
        context: RiskCheckContext,
        warnings: List[str],
        risk_metrics: dict,
    ) -> Optional[RiskCheckResult]:
        """Check if volatility is within acceptable range."""
        symbol = context.symbol
        price = context.price
        atr = context.atr

        # Get symbol-specific max ATR if configured
        max_atr_pct = self.settings.get_symbol_config(
            symbol, "max_atr_pct", self.settings.trade_risk.max_atr_pct
        )

        if atr is None:
            # Try to get ATR percentage directly
            atr_pct = context.atr_pct
            if atr_pct is None:
                warnings.append("ATR not available for volatility check")
                return None
        else:
            atr_pct = atr / price if price > 0 else 0

        risk_metrics["atr_pct"] = atr_pct
        risk_metrics["max_atr_pct"] = max_atr_pct

        if atr_pct > max_atr_pct:
            return self._reject(
                f"Volatility too high: ATR {atr_pct:.1%} > {max_atr_pct:.1%}",
                risk_score=0.85,
                context=context,
                risk_metrics=risk_metrics,
            )

        if atr_pct > max_atr_pct * 0.8:
            warnings.append(f"Elevated volatility: ATR {atr_pct:.1%}")

        return None

    def _check_trend(
        self,
        context: RiskCheckContext,
        warnings: List[str],
        risk_metrics: dict,
    ) -> Optional[RiskCheckResult]:
        """Check if price is above 200 SMA for BUY signals."""
        symbol = context.symbol
        price = context.price

        # Check if trend filter is enabled for this symbol
        require_sma = self.settings.get_symbol_config(
            symbol,
            "require_above_sma200",
            self.settings.trade_risk.require_above_sma200,
        )

        if not require_sma:
            risk_metrics["trend_filter"] = "disabled"
            return None

        sma_200 = context.sma_200
        if sma_200 is None:
            warnings.append("SMA 200 not available for trend check")
            return None

        risk_metrics["sma_200"] = sma_200
        risk_metrics["price_vs_sma"] = (price - sma_200) / sma_200 if sma_200 > 0 else 0

        if price < sma_200:
            return self._reject(
                f"Price ${price:.2f} below SMA 200 (${sma_200:.2f})",
                risk_score=0.70,
                context=context,
                risk_metrics=risk_metrics,
            )

        # Check how far above
        pct_above = (price - sma_200) / sma_200
        if pct_above > 0.20:
            warnings.append(
                f"Price {pct_above:.1%} above SMA 200 - potential overextension"
            )

        return None

    def _check_liquidity(
        self,
        context: RiskCheckContext,
        warnings: List[str],
        risk_metrics: dict,
    ) -> Optional[RiskCheckResult]:
        """Check if current volume meets minimum threshold."""
        volume = context.volume
        avg_volume = context.avg_volume
        min_volume_pct = self.settings.trade_risk.min_volume_pct

        # Try to get average volume from market data if not in indicators
        if avg_volume is None and self.market_data is not None:
            avg_volume = self.market_data.get_average_volume(
                context.symbol, lookback_days=20
            )

        if volume is None or avg_volume is None:
            warnings.append("Volume data not available for liquidity check")
            return None

        if avg_volume <= 0:
            warnings.append("Invalid average volume data")
            return None

        volume_ratio = volume / avg_volume
        risk_metrics["volume"] = volume
        risk_metrics["avg_volume"] = avg_volume
        risk_metrics["volume_ratio"] = volume_ratio

        if volume_ratio < min_volume_pct:
            return self._reject(
                f"Low liquidity: volume {volume_ratio:.1%} of average "
                f"(min {min_volume_pct:.1%})",
                risk_score=0.65,
                context=context,
                risk_metrics=risk_metrics,
            )

        if volume_ratio < min_volume_pct * 1.5:
            warnings.append(f"Below-average volume: {volume_ratio:.1%} of average")

        return None

    def _calculate_trade_risk(self, risk_metrics: dict) -> float:
        """Calculate aggregate trade risk score."""
        scores = []

        # Volatility contribution
        atr_pct = risk_metrics.get("atr_pct", 0)
        max_atr_pct = risk_metrics.get("max_atr_pct", 0.05)
        if max_atr_pct > 0:
            scores.append(atr_pct / max_atr_pct)

        # Trend contribution (inverse - further above SMA = lower risk)
        price_vs_sma = risk_metrics.get("price_vs_sma", 0)
        if price_vs_sma > 0:
            # Above SMA reduces risk, capped at 0.2
            scores.append(max(0, 0.5 - min(price_vs_sma, 0.20)))
        else:
            scores.append(0.8)  # Below SMA = higher risk

        # Liquidity contribution (inverse - higher volume = lower risk)
        volume_ratio = risk_metrics.get("volume_ratio", 1.0)
        if volume_ratio > 0:
            scores.append(1.0 / (1 + volume_ratio))

        if not scores:
            return 0.5

        return sum(scores) / len(scores)
