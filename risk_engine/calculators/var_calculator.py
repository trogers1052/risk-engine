"""
VaR and CVaR calculator using Riskfolio-Lib.

Provides multiple VaR calculation methods:
- Historical simulation
- Parametric (variance-covariance)
- Riskfolio-Lib advanced methods (when available)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import RiskSettings

logger = logging.getLogger(__name__)

# Try to import Riskfolio-Lib
try:
    import riskfolio as rp

    HAS_RISKFOLIO = True
except ImportError:
    HAS_RISKFOLIO = False
    logger.info("Riskfolio-Lib not installed, using fallback methods")

# Try to import skfolio
try:
    from skfolio.risk_measure import CVaR, VaR

    HAS_SKFOLIO = True
except ImportError:
    HAS_SKFOLIO = False


@dataclass
class VaRResult:
    """Result of VaR calculation."""

    var: float
    cvar: float
    confidence: float
    method: str
    volatility: Optional[float] = None
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    additional_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


class VaRCalculator:
    """
    Calculate Value at Risk and related risk metrics.

    Uses Riskfolio-Lib for sophisticated calculations when available,
    with fallback to historical simulation.
    """

    def __init__(self, settings: RiskSettings):
        self.settings = settings

    def calculate_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence: Optional[float] = None,
        method: str = "historical",
    ) -> VaRResult:
        """
        Calculate VaR and CVaR for a portfolio.

        Args:
            returns: DataFrame of asset returns (columns = assets)
            weights: Array of portfolio weights
            confidence: Confidence level (default from config)
            method: Calculation method ("historical", "parametric", "riskfolio")

        Returns:
            VaRResult with VaR, CVaR, and additional metrics
        """
        if confidence is None:
            confidence = self.settings.var.confidence

        # Calculate portfolio returns
        port_returns = (returns * weights).sum(axis=1)

        if method == "riskfolio" and HAS_RISKFOLIO:
            return self._riskfolio_var(returns, weights, confidence)
        elif method == "parametric":
            return self._parametric_var(port_returns, confidence)
        else:
            return self._historical_var(port_returns, confidence)

    def calculate_symbol_var(
        self,
        returns: pd.Series,
        confidence: Optional[float] = None,
    ) -> VaRResult:
        """Calculate VaR for a single symbol."""
        if confidence is None:
            confidence = self.settings.var.confidence

        return self._historical_var(returns, confidence)

    def _historical_var(
        self,
        port_returns: pd.Series,
        confidence: float,
    ) -> VaRResult:
        """Calculate VaR using historical simulation."""
        alpha = 1 - confidence
        percentile = alpha * 100

        # VaR: percentile of losses
        var = abs(np.percentile(port_returns, percentile))

        # CVaR: mean of returns below VaR threshold
        var_threshold = np.percentile(port_returns, percentile)
        tail_returns = port_returns[port_returns <= var_threshold]
        cvar = abs(tail_returns.mean()) if len(tail_returns) > 0 else var

        # Additional metrics
        volatility = port_returns.std() * np.sqrt(252)
        mean_return = port_returns.mean() * 252

        # Sharpe (assuming 0 risk-free rate)
        sharpe = mean_return / volatility if volatility > 0 else 0

        # Max drawdown
        cumulative = (1 + port_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())

        return VaRResult(
            var=var,
            cvar=cvar,
            confidence=confidence,
            method="historical",
            volatility=volatility,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            additional_metrics={
                "mean_return": mean_return,
                "skewness": port_returns.skew(),
                "kurtosis": port_returns.kurtosis(),
            },
        )

    def _parametric_var(
        self,
        port_returns: pd.Series,
        confidence: float,
    ) -> VaRResult:
        """Calculate VaR using parametric (variance-covariance) method."""
        from scipy import stats

        mean = port_returns.mean()
        std = port_returns.std()

        # Z-score for confidence level
        z = stats.norm.ppf(1 - confidence)

        # VaR
        var = abs(mean + z * std)

        # CVaR for normal distribution
        pdf_z = stats.norm.pdf(z)
        cvar = abs(mean + std * pdf_z / (1 - confidence))

        volatility = std * np.sqrt(252)
        mean_return = mean * 252
        sharpe = mean_return / volatility if volatility > 0 else 0

        return VaRResult(
            var=var,
            cvar=cvar,
            confidence=confidence,
            method="parametric",
            volatility=volatility,
            sharpe=sharpe,
            additional_metrics={
                "mean_return": mean_return,
                "daily_std": std,
            },
        )

    def _riskfolio_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence: float,
    ) -> VaRResult:
        """Calculate VaR using Riskfolio-Lib."""
        try:
            alpha = 1 - confidence
            port_returns = (returns * weights).sum(axis=1)

            # Riskfolio VaR/CVaR functions
            var = abs(rp.RiskFunctions.VaR_Hist(port_returns.values, alpha))
            cvar = abs(rp.RiskFunctions.CVaR_Hist(port_returns.values, alpha))

            # Additional risk measures from Riskfolio
            std = port_returns.std()
            volatility = std * np.sqrt(252)
            mean_return = port_returns.mean() * 252
            sharpe = mean_return / volatility if volatility > 0 else 0

            # Additional metrics
            additional = {}

            # Max drawdown
            cumulative = (1 + port_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())

            # Sortino ratio (downside deviation)
            negative_returns = port_returns[port_returns < 0]
            downside_std = negative_returns.std() * np.sqrt(252)
            additional["sortino"] = (
                mean_return / downside_std if downside_std > 0 else 0
            )

            # Semi-deviation
            additional["semi_deviation"] = downside_std

            return VaRResult(
                var=var,
                cvar=cvar,
                confidence=confidence,
                method="riskfolio",
                volatility=volatility,
                sharpe=sharpe,
                max_drawdown=max_drawdown,
                additional_metrics=additional,
            )

        except Exception as e:
            logger.warning(f"Riskfolio calculation failed: {e}, falling back")
            port_returns = (returns * weights).sum(axis=1)
            return self._historical_var(port_returns, confidence)

    def calculate_portfolio_risk_metrics(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio risk metrics.

        Args:
            returns: DataFrame of asset returns
            weights: Array of portfolio weights

        Returns:
            Dictionary of risk metrics
        """
        port_returns = (returns * weights).sum(axis=1)

        metrics = {}

        # Basic statistics
        metrics["mean_return_daily"] = port_returns.mean()
        metrics["mean_return_annual"] = port_returns.mean() * 252
        metrics["volatility_daily"] = port_returns.std()
        metrics["volatility_annual"] = port_returns.std() * np.sqrt(252)

        # VaR/CVaR at multiple confidence levels
        for conf in [0.95, 0.99]:
            alpha = 1 - conf
            percentile = alpha * 100
            var = abs(np.percentile(port_returns, percentile))
            var_threshold = np.percentile(port_returns, percentile)
            tail = port_returns[port_returns <= var_threshold]
            cvar = abs(tail.mean()) if len(tail) > 0 else var

            metrics[f"var_{int(conf*100)}"] = var
            metrics[f"cvar_{int(conf*100)}"] = cvar

        # Sharpe ratio
        vol = metrics["volatility_annual"]
        metrics["sharpe"] = metrics["mean_return_annual"] / vol if vol > 0 else 0

        # Sortino ratio
        negative_returns = port_returns[port_returns < 0]
        downside_vol = negative_returns.std() * np.sqrt(252)
        metrics["sortino"] = (
            metrics["mean_return_annual"] / downside_vol if downside_vol > 0 else 0
        )

        # Max drawdown
        cumulative = (1 + port_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        metrics["max_drawdown"] = abs(drawdowns.min())

        # Skewness and kurtosis
        metrics["skewness"] = port_returns.skew()
        metrics["kurtosis"] = port_returns.kurtosis()

        return metrics

    def calculate_marginal_var(
        self,
        returns: pd.DataFrame,
        current_weights: Dict[str, float],
        new_symbol: str,
        new_weight: float,
        confidence: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculate marginal VaR contribution of adding a new position.

        Returns:
            Tuple of (current_var, new_var)
        """
        if confidence is None:
            confidence = self.settings.var.confidence

        # Get symbols in returns
        available_symbols = [
            s for s in current_weights.keys() if s in returns.columns
        ]

        if not available_symbols:
            return 0.0, 0.0

        # Current portfolio VaR
        current_weight_array = np.array([
            current_weights[s] for s in available_symbols
        ])
        current_returns = returns[available_symbols]
        current_port_returns = (current_returns * current_weight_array).sum(axis=1)
        alpha = 1 - confidence
        current_var = abs(np.percentile(current_port_returns, alpha * 100))

        # New portfolio VaR (if new symbol has data)
        if new_symbol in returns.columns:
            # Scale down existing weights
            scale = 1 - new_weight
            new_symbols = available_symbols + [new_symbol]
            new_weight_array = np.array([
                current_weights[s] * scale for s in available_symbols
            ] + [new_weight])

            new_returns = returns[new_symbols]
            new_port_returns = (new_returns * new_weight_array).sum(axis=1)
            new_var = abs(np.percentile(new_port_returns, alpha * 100))
        else:
            new_var = current_var

        return current_var, new_var
