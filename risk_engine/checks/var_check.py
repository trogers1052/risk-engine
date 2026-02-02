"""
Value at Risk (VaR) and Conditional VaR (CVaR) risk check.

Uses Riskfolio-Lib for risk calculations when available.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import RiskSettings
from ..data.market_data import MarketDataLoader
from ..models.risk_result import RiskCheckResult
from .base import RiskCheck, RiskCheckContext

logger = logging.getLogger(__name__)

# Try to import Riskfolio-Lib
try:
    import riskfolio as rp

    HAS_RISKFOLIO = True
except ImportError:
    HAS_RISKFOLIO = False
    logger.warning("Riskfolio-Lib not available, using fallback VaR calculation")


class VaRCheck(RiskCheck):
    """
    Value at Risk and Conditional VaR check.

    Uses Riskfolio-Lib when available for sophisticated risk measures.
    Falls back to historical simulation otherwise.

    Checks:
    - Portfolio VaR at 95% confidence < 5% daily
    - Portfolio CVaR at 95% confidence < 8% daily
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
        return "var_check"

    @property
    def description(self) -> str:
        return "Check Value at Risk and Conditional VaR limits"

    def can_check(self, context: RiskCheckContext) -> bool:
        """Need market data and portfolio for VaR calculation."""
        if not self.settings.var.enabled:
            return False
        if self.market_data is None:
            return False
        if context.portfolio is None or len(context.portfolio.symbols) == 0:
            # First position - can check individual VaR
            return True
        return True

    def check(self, context: RiskCheckContext) -> RiskCheckResult:
        """Calculate portfolio VaR with proposed position."""
        warnings: List[str] = []
        risk_metrics: Dict[str, float] = {}

        portfolio = context.portfolio
        symbol = context.symbol
        price = context.price

        # If no existing positions, check individual symbol VaR
        if portfolio is None or len(portfolio.symbols) == 0:
            return self._check_symbol_var(context, warnings, risk_metrics)

        # Get current weights
        current_weights = portfolio.get_weights()

        # Estimate new position weight (use a default allocation)
        # The actual size will be determined by position sizing check
        estimated_new_weight = min(
            0.10,  # Assume 10% allocation for VaR calculation
            self.settings.get_max_position_pct(symbol),
        )

        # Create hypothetical portfolio with new position
        new_weights = self._create_hypothetical_weights(
            current_weights, symbol, estimated_new_weight
        )

        # Calculate VaR and CVaR
        var_result = self._calculate_portfolio_var(
            new_weights, warnings, risk_metrics
        )

        if var_result is None:
            # Could not calculate VaR
            warnings.append("Could not calculate portfolio VaR")
            return self._approve(
                recommended_shares=0,
                max_shares=0,
                recommended_dollar_amount=0,
                risk_score=0.5,
                context=context,
                reason="VaR check skipped - insufficient data",
                warnings=warnings,
                risk_metrics=risk_metrics,
            )

        var, cvar = var_result
        risk_metrics["var_95"] = var
        risk_metrics["cvar_95"] = cvar

        # Check limits
        max_var = self.settings.var.max_var_daily_pct
        max_cvar = self.settings.var.max_cvar_daily_pct

        if var > max_var:
            return self._reject(
                f"Portfolio VaR {var:.2%} exceeds limit {max_var:.2%}",
                risk_score=0.9,
                context=context,
                risk_metrics=risk_metrics,
            )

        if cvar > max_cvar:
            return self._reject(
                f"Portfolio CVaR {cvar:.2%} exceeds limit {max_cvar:.2%}",
                risk_score=0.85,
                context=context,
                risk_metrics=risk_metrics,
            )

        # Warnings for elevated risk
        if var > max_var * 0.8:
            warnings.append(f"Elevated VaR: {var:.2%} (limit: {max_var:.2%})")
        if cvar > max_cvar * 0.8:
            warnings.append(f"Elevated CVaR: {cvar:.2%} (limit: {max_cvar:.2%})")

        # Risk score based on VaR/CVaR utilization
        risk_score = max(var / max_var, cvar / max_cvar)

        return self._approve(
            recommended_shares=0,
            max_shares=0,
            recommended_dollar_amount=0,
            risk_score=risk_score,
            context=context,
            reason=f"VaR {var:.2%}, CVaR {cvar:.2%} within limits",
            warnings=warnings,
            risk_metrics=risk_metrics,
        )

    def _check_symbol_var(
        self,
        context: RiskCheckContext,
        warnings: List[str],
        risk_metrics: dict,
    ) -> RiskCheckResult:
        """Check VaR for a single symbol (no portfolio)."""
        symbol = context.symbol

        returns = self.market_data.get_daily_returns(
            symbol, self.settings.var.lookback_days
        )

        if returns is None or len(returns) < 30:
            warnings.append(f"Insufficient return data for {symbol}")
            return self._approve(
                recommended_shares=0,
                max_shares=0,
                recommended_dollar_amount=0,
                risk_score=0.5,
                context=context,
                reason="VaR check skipped - insufficient data",
                warnings=warnings,
                risk_metrics=risk_metrics,
            )

        # Calculate historical VaR/CVaR
        confidence = self.settings.var.confidence
        var = self._historical_var(returns, confidence)
        cvar = self._historical_cvar(returns, confidence)

        risk_metrics["symbol_var_95"] = var
        risk_metrics["symbol_cvar_95"] = cvar

        max_var = self.settings.var.max_var_daily_pct

        if var > max_var:
            return self._reject(
                f"Symbol VaR {var:.2%} exceeds portfolio limit {max_var:.2%}",
                risk_score=0.85,
                context=context,
                risk_metrics=risk_metrics,
            )

        return self._approve(
            recommended_shares=0,
            max_shares=0,
            recommended_dollar_amount=0,
            risk_score=var / max_var,
            context=context,
            reason=f"Symbol VaR {var:.2%} acceptable",
            warnings=warnings,
            risk_metrics=risk_metrics,
        )

    def _create_hypothetical_weights(
        self,
        current_weights: Dict[str, float],
        new_symbol: str,
        new_weight: float,
    ) -> Dict[str, float]:
        """Create hypothetical portfolio weights with new position."""
        # Scale down existing weights to make room
        scale_factor = 1.0 - new_weight

        new_weights = {
            symbol: weight * scale_factor
            for symbol, weight in current_weights.items()
        }
        new_weights[new_symbol] = new_weight

        return new_weights

    def _calculate_portfolio_var(
        self,
        weights: Dict[str, float],
        warnings: List[str],
        risk_metrics: dict,
    ) -> Optional[tuple]:
        """Calculate portfolio VaR and CVaR."""
        symbols = list(weights.keys())

        # Get returns data
        returns_df = self.market_data.get_multiple_returns(
            symbols, self.settings.var.lookback_days
        )

        if returns_df is None or len(returns_df) < 30:
            return None

        # Filter to symbols with data
        available_symbols = [s for s in symbols if s in returns_df.columns]
        if not available_symbols:
            return None

        # Normalize weights for available symbols
        weight_sum = sum(weights.get(s, 0) for s in available_symbols)
        if weight_sum <= 0:
            return None

        weight_array = np.array([
            weights.get(s, 0) / weight_sum for s in available_symbols
        ])

        returns_matrix = returns_df[available_symbols].values

        if HAS_RISKFOLIO:
            return self._riskfolio_var(
                returns_matrix, weight_array, warnings, risk_metrics
            )
        else:
            return self._fallback_var(
                returns_matrix, weight_array, warnings
            )

    def _riskfolio_var(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        warnings: List[str],
        risk_metrics: dict,
    ) -> Optional[tuple]:
        """Calculate VaR using Riskfolio-Lib."""
        try:
            returns_df = pd.DataFrame(returns)
            alpha = 1 - self.settings.var.confidence

            # Calculate portfolio returns
            port_returns = returns_df @ weights

            # Historical VaR
            var = rp.RiskFunctions.VaR_Hist(port_returns.values, alpha)

            # Historical CVaR
            cvar = rp.RiskFunctions.CVaR_Hist(port_returns.values, alpha)

            # Additional metrics from Riskfolio
            risk_metrics["volatility"] = float(np.std(port_returns) * np.sqrt(252))

            return abs(var), abs(cvar)

        except Exception as e:
            warnings.append(f"Riskfolio calculation error: {e}")
            logger.error(f"Riskfolio VaR calculation failed: {e}")
            return self._fallback_var(returns, weights, warnings)

    def _fallback_var(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        warnings: List[str],
    ) -> Optional[tuple]:
        """Fallback VaR calculation using historical simulation."""
        try:
            # Calculate portfolio returns
            port_returns = returns @ weights

            confidence = self.settings.var.confidence
            var = self._historical_var(port_returns, confidence)
            cvar = self._historical_cvar(port_returns, confidence)

            warnings.append("Using fallback VaR calculation")
            return var, cvar

        except Exception as e:
            logger.error(f"Fallback VaR calculation failed: {e}")
            return None

    @staticmethod
    def _historical_var(returns, confidence: float) -> float:
        """Calculate historical VaR."""
        if isinstance(returns, pd.Series):
            returns = returns.values
        percentile = (1 - confidence) * 100
        return abs(np.percentile(returns, percentile))

    @staticmethod
    def _historical_cvar(returns, confidence: float) -> float:
        """Calculate historical CVaR (Expected Shortfall)."""
        if isinstance(returns, pd.Series):
            returns = returns.values
        percentile = (1 - confidence) * 100
        var_threshold = np.percentile(returns, percentile)
        tail_returns = returns[returns <= var_threshold]
        if len(tail_returns) == 0:
            return abs(var_threshold)
        return abs(np.mean(tail_returns))
