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
    position count, portfolio heat, and drawdown circuit breaker.

    Checks (in order):
    1. Max open positions (default 5)
    2. Portfolio heat — total risk across all positions (default 8%)
    3. Max total exposure / min cash buffer (default 95%/5%)
    4. Max concentration per symbol (default 20%)
    5. Sector concentration — position count (default 2) + % exposure (default 40%)
    6. Max correlation with existing positions (default 80%)
    7. Drawdown circuit breaker (reduces sizing at 10%, halts entries at 15%)
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

    # Regime-aware limit adjustments.  In volatile or crisis markets,
    # we tighten portfolio limits to reduce exposure.  In bear markets,
    # we reduce position count.  These multiply against the configured
    # base limits — they never loosen limits beyond the config defaults.
    REGIME_ADJUSTMENTS: dict = {
        "BULL": {},  # no changes
        "SIDEWAYS": {},  # no changes
        "BEAR": {
            "max_open_positions_mult": 0.80,       # 5→4
            "max_portfolio_heat_mult": 0.75,        # 8%→6%
        },
        "VOLATILE": {
            "max_open_positions_mult": 0.60,       # 5→3
            "max_portfolio_heat_mult": 0.625,       # 8%→5%
            "max_correlation_mult": 0.75,           # 0.80→0.60
            "max_sector_positions_mult": 0.50,      # 2→1
        },
        "CRISIS": {
            "max_open_positions_mult": 0.40,       # 5→2
            "max_portfolio_heat_mult": 0.50,        # 8%→4%
            "max_correlation_mult": 0.625,          # 0.80→0.50
            "max_sector_positions_mult": 0.50,      # 2→1
        },
    }

    def _get_regime(self) -> str:
        """Read current market regime from Redis via portfolio manager."""
        if self.portfolio_manager is not None:
            return self.portfolio_manager.get_regime()
        return "UNKNOWN"

    def _apply_regime_adjustments(
        self, regime: str, warnings: List[str], risk_metrics: dict
    ) -> None:
        """Apply regime-aware multipliers to settings for this check run.

        Stores adjusted values in risk_metrics so individual checks can read
        them instead of accessing self.settings directly for the adjusted fields.
        """
        adjustments = self.REGIME_ADJUSTMENTS.get(regime, {})
        base = self.settings.portfolio_risk

        risk_metrics["regime"] = regime
        risk_metrics["effective_max_positions"] = int(
            base.max_open_positions
            * adjustments.get("max_open_positions_mult", 1.0)
        )
        risk_metrics["effective_max_heat"] = (
            base.max_portfolio_heat_pct
            * adjustments.get("max_portfolio_heat_mult", 1.0)
        )
        risk_metrics["effective_max_correlation"] = (
            base.max_correlation
            * adjustments.get("max_correlation_mult", 1.0)
        )
        risk_metrics["effective_max_sector_positions"] = int(
            base.max_sector_positions
            * adjustments.get("max_sector_positions_mult", 1.0)
        )
        # Ensure minimums
        risk_metrics["effective_max_positions"] = max(
            1, risk_metrics["effective_max_positions"]
        )
        risk_metrics["effective_max_sector_positions"] = max(
            1, risk_metrics["effective_max_sector_positions"]
        )

        if adjustments:
            warnings.append(
                f"Regime-adjusted limits ({regime}): "
                f"positions={risk_metrics['effective_max_positions']}, "
                f"heat={risk_metrics['effective_max_heat']:.1%}, "
                f"correlation={risk_metrics['effective_max_correlation']:.2f}"
            )

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

        # Apply regime-aware limit adjustments
        regime = self._get_regime()
        self._apply_regime_adjustments(regime, warnings, risk_metrics)

        # Check 1: Max open positions
        pos_result = self._check_open_positions(context, warnings, risk_metrics)
        if pos_result is not None:
            return pos_result

        # Check 2: Portfolio heat (total risk across all positions)
        heat_result = self._check_portfolio_heat(context, warnings, risk_metrics)
        if heat_result is not None:
            return heat_result

        # Check 3: Total exposure limit
        exposure_result = self._check_exposure(context, warnings, risk_metrics)
        if exposure_result is not None:
            return exposure_result

        # Check 4: Concentration limit
        concentration_result = self._check_concentration(
            context, warnings, risk_metrics
        )
        if concentration_result is not None:
            return concentration_result

        # Check 5: Sector concentration (position count + % exposure)
        sector_result = self._check_sector_concentration(
            context, warnings, risk_metrics
        )
        if sector_result is not None:
            return sector_result

        # Check 6: Correlation with existing positions
        correlation_result = self._check_correlation(
            context, warnings, risk_metrics
        )
        if correlation_result is not None:
            return correlation_result

        # Check 7: Drawdown circuit breaker (reduces sizing at 10%, halts at 15%)
        drawdown_result = self._check_drawdown_circuit_breaker(
            context, warnings, risk_metrics
        )
        if drawdown_result is not None:
            return drawdown_result

        # Check 8: Capital temperature (market stress → size reduction)
        self._check_capital_temperature(context, warnings, risk_metrics)

        # All checks passed - calculate aggregate risk score
        risk_score = self._calculate_aggregate_risk(risk_metrics)

        # Log portfolio state summary at flow boundary
        logger.info(
            f"Portfolio risk passed for {symbol}: "
            f"heat={risk_metrics.get('portfolio_heat', 0):.1%}, "
            f"exposure={risk_metrics.get('current_exposure', 0):.1%}, "
            f"drawdown={risk_metrics.get('drawdown_pct', 0):.1%}, "
            f"positions={risk_metrics.get('open_positions', 0)}/"
            f"{risk_metrics.get('max_positions', 0)}"
        )

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
        max_positions = risk_metrics.get(
            "effective_max_positions",
            self.settings.portfolio_risk.max_open_positions,
        )
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

    def _check_portfolio_heat(
        self,
        context: RiskCheckContext,
        warnings: List[str],
        risk_metrics: dict,
    ) -> Optional[RiskCheckResult]:
        """Check total portfolio risk (heat).

        Portfolio heat = sum of risk_per_trade_pct for each open position.
        Each position was originally sized for its configured risk_per_trade_pct,
        so heat approximates the total capital at risk if all stops hit
        simultaneously.
        """
        max_heat = risk_metrics.get(
            "effective_max_heat",
            self.settings.portfolio_risk.max_portfolio_heat_pct,
        )
        portfolio = context.portfolio

        # Sum the configured risk_per_trade for each open position
        total_heat = 0.0
        for symbol in portfolio.symbols:
            total_heat += self.settings.get_risk_per_trade_pct(symbol)

        risk_metrics["portfolio_heat"] = total_heat
        risk_metrics["max_portfolio_heat"] = max_heat

        # Would adding this new position exceed heat limit?
        new_position_risk = self.settings.get_risk_per_trade_pct(context.symbol)
        projected_heat = total_heat + new_position_risk

        if projected_heat > max_heat:
            return self._reject(
                f"Portfolio heat {projected_heat:.1%} would exceed "
                f"limit {max_heat:.1%} (current: {total_heat:.1%}, "
                f"new: {new_position_risk:.1%} for {context.symbol}, "
                f"{portfolio.position_count} positions open)",
                risk_score=0.85,
                context=context,
                risk_metrics=risk_metrics,
            )

        remaining = max_heat - projected_heat
        if remaining < 0.01:
            warnings.append(
                f"Portfolio heat near limit: {projected_heat:.1%}/{max_heat:.1%}"
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
        """Check sector position count and exposure limit.

        Two checks in order:
        1. Position count — max N positions per sector (default 2)
        2. Percent exposure — max % of equity in one sector (default 40%)
        """
        symbol = context.symbol
        sector = self.settings.get_sector_for_symbol(symbol)

        if sector is None:
            # Symbol not in any sector group — can't check, pass with warning
            warnings.append(f"No sector mapping for {symbol}")
            return None

        portfolio = context.portfolio
        sector_symbols = self.settings.sector_groups.get(sector, [])

        held_in_sector = [
            s for s in sector_symbols if portfolio.has_position(s)
        ]
        risk_metrics["sector"] = sector
        risk_metrics["sector_positions"] = len(held_in_sector)

        # Check position count limit
        max_sector_positions = risk_metrics.get(
            "effective_max_sector_positions",
            self.settings.portfolio_risk.max_sector_positions,
        )
        # Don't count the candidate symbol if already held (scale-in)
        effective_count = len(held_in_sector)
        if symbol in held_in_sector:
            effective_count -= 1

        if effective_count >= max_sector_positions:
            return self._reject(
                f"Sector '{sector}' at position limit: "
                f"{len(held_in_sector)}/{max_sector_positions} "
                f"(held: {held_in_sector})",
                risk_score=0.80,
                context=context,
                risk_metrics=risk_metrics,
            )

        # Check % exposure
        max_sector_pct = self.settings.portfolio_risk.max_sector_exposure_pct
        sector_weight = sum(
            portfolio.get_position_weight(s) for s in sector_symbols
        )
        risk_metrics["sector_exposure"] = sector_weight
        risk_metrics["max_sector_exposure"] = max_sector_pct

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
        max_correlation = risk_metrics.get(
            "effective_max_correlation",
            self.settings.portfolio_risk.max_correlation,
        )

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
    ) -> Optional[RiskCheckResult]:
        """Check drawdown from peak equity.

        Two thresholds:
        - drawdown_threshold_pct (default 10%): reduce position sizes
        - drawdown_halt_pct (default 15%): block all new entries
        """
        portfolio = context.portfolio
        threshold = self.settings.portfolio_risk.drawdown_threshold_pct
        halt_pct = self.settings.portfolio_risk.drawdown_halt_pct
        current_equity = portfolio.total_equity

        if current_equity <= 0:
            return None

        # Get peak equity (updates Redis if new high)
        if self.portfolio_manager is not None:
            peak_equity = self.portfolio_manager.get_peak_equity(current_equity)
        else:
            peak_equity = current_equity

        if peak_equity <= 0:
            return None

        drawdown = (peak_equity - current_equity) / peak_equity
        risk_metrics["peak_equity"] = peak_equity
        risk_metrics["drawdown_pct"] = drawdown
        risk_metrics["drawdown_threshold"] = threshold
        risk_metrics["drawdown_halt_pct"] = halt_pct

        # Hard halt at higher threshold
        if drawdown >= halt_pct:
            return self._reject(
                f"DRAWDOWN HALT: {drawdown:.1%} from peak ${peak_equity:.2f} "
                f"exceeds {halt_pct:.0%} limit. "
                f"All new entries blocked until equity recovers.",
                risk_score=0.95,
                context=context,
                risk_metrics=risk_metrics,
            )

        # Size reduction at lower threshold
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

        return None

    def _check_capital_temperature(
        self,
        context: RiskCheckContext,
        warnings: List[str],
        risk_metrics: dict,
    ) -> None:
        """Compute capital temperature from market stress indicators.

        Temperature ∈ [0, 1] based on VIX and HY credit spread.
        Sets context.metadata["capital_temperature_reduction"] for position
        sizing to apply.  Never rejects — only reduces position sizes.
        """
        config = self.settings.capital_temperature
        if not config.enabled or self.portfolio_manager is None:
            return

        macro = self.portfolio_manager.get_macro_signals()
        if not isinstance(macro, dict) or not macro:
            warnings.append("Capital temperature skipped: no macro data")
            return

        vix = macro.get("vix")
        hy = macro.get("hy_spread")

        # Score each factor: 0 = calm, 1 = crisis
        vix_score = 0.0
        if vix is not None:
            vix_range = config.vix_ceiling - config.vix_floor
            if vix_range > 0:
                vix_score = max(0.0, min(1.0,
                    (vix - config.vix_floor) / vix_range
                ))

        hy_score = 0.0
        if hy is not None:
            hy_range = config.hy_ceiling - config.hy_floor
            if hy_range > 0:
                hy_score = max(0.0, min(1.0,
                    (hy - config.hy_floor) / hy_range
                ))

        # Weighted temperature
        temperature = (
            config.vix_weight * vix_score
            + config.hy_spread_weight * hy_score
        )
        reduction = temperature * config.max_reduction

        risk_metrics["capital_temperature"] = temperature
        risk_metrics["capital_temp_vix_score"] = vix_score
        risk_metrics["capital_temp_hy_score"] = hy_score
        risk_metrics["capital_temp_reduction"] = reduction

        if reduction > 0.01:
            context.metadata["capital_temperature_reduction"] = reduction
            warnings.append(
                f"Capital temperature {temperature:.2f} "
                f"(VIX={vix or 'N/A'}, HY={hy or 'N/A'}): "
                f"position sizes reduced by {reduction:.0%}"
            )

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
