"""
Risk metrics calculator using empyrical-reloaded.

Provides quick calculation of common performance metrics:
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Calmar ratio
- And more...
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import empyrical-reloaded
try:
    import empyrical

    HAS_EMPYRICAL = True
except ImportError:
    HAS_EMPYRICAL = False
    logger.info("empyrical-reloaded not installed, using fallback calculations")


@dataclass
class PerformanceMetrics:
    """Collection of performance metrics."""

    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    annual_return: float
    annual_volatility: float
    calmar_ratio: float
    omega_ratio: Optional[float] = None
    tail_ratio: Optional[float] = None
    value_at_risk: Optional[float] = None
    conditional_var: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "annual_return": round(self.annual_return, 4),
            "annual_volatility": round(self.annual_volatility, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "omega_ratio": round(self.omega_ratio, 4) if self.omega_ratio else None,
            "tail_ratio": round(self.tail_ratio, 4) if self.tail_ratio else None,
            "value_at_risk": round(self.value_at_risk, 4) if self.value_at_risk else None,
            "conditional_var": (
                round(self.conditional_var, 4) if self.conditional_var else None
            ),
        }


class MetricsCalculator:
    """
    Calculate performance and risk metrics.

    Uses empyrical-reloaded when available for comprehensive metrics.
    Falls back to manual calculations otherwise.
    """

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 0)
        """
        self.risk_free_rate = risk_free_rate
        self._daily_rf = risk_free_rate / 252

    def calculate_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Args:
            returns: Daily returns series
            benchmark_returns: Optional benchmark returns

        Returns:
            PerformanceMetrics dataclass
        """
        if HAS_EMPYRICAL:
            return self._empyrical_metrics(returns, benchmark_returns)
        else:
            return self._fallback_metrics(returns)

    def _empyrical_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series],
    ) -> PerformanceMetrics:
        """Calculate metrics using empyrical-reloaded."""
        try:
            sharpe = empyrical.sharpe_ratio(
                returns, risk_free=self._daily_rf
            )
            sortino = empyrical.sortino_ratio(
                returns, required_return=self._daily_rf
            )
            max_dd = empyrical.max_drawdown(returns)
            annual_ret = empyrical.annual_return(returns)
            annual_vol = empyrical.annual_volatility(returns)
            calmar = empyrical.calmar_ratio(returns)

            # Optional metrics
            try:
                omega = empyrical.omega_ratio(
                    returns, risk_free=self._daily_rf
                )
            except Exception:
                omega = None

            try:
                tail = empyrical.tail_ratio(returns)
            except Exception:
                tail = None

            # VaR and CVaR
            var = self._calculate_var(returns, 0.95)
            cvar = self._calculate_cvar(returns, 0.95)

            return PerformanceMetrics(
                sharpe_ratio=float(sharpe) if not np.isnan(sharpe) else 0,
                sortino_ratio=float(sortino) if not np.isnan(sortino) else 0,
                max_drawdown=float(abs(max_dd)) if not np.isnan(max_dd) else 0,
                annual_return=float(annual_ret) if not np.isnan(annual_ret) else 0,
                annual_volatility=float(annual_vol) if not np.isnan(annual_vol) else 0,
                calmar_ratio=float(calmar) if not np.isnan(calmar) else 0,
                omega_ratio=float(omega) if omega and not np.isnan(omega) else None,
                tail_ratio=float(tail) if tail and not np.isnan(tail) else None,
                value_at_risk=var,
                conditional_var=cvar,
            )

        except Exception as e:
            logger.warning(f"empyrical calculation error: {e}")
            return self._fallback_metrics(returns)

    def _fallback_metrics(self, returns: pd.Series) -> PerformanceMetrics:
        """Calculate metrics without empyrical."""
        # Annual return
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Annual volatility
        annual_vol = returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0

        # Sortino ratio
        negative_returns = returns[returns < self._daily_rf]
        downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino = (annual_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0

        # Calmar ratio
        calmar = annual_return / max_dd if max_dd > 0 else 0

        # VaR and CVaR
        var = self._calculate_var(returns, 0.95)
        cvar = self._calculate_cvar(returns, 0.95)

        return PerformanceMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            annual_return=annual_return,
            annual_volatility=annual_vol,
            calmar_ratio=calmar,
            value_at_risk=var,
            conditional_var=cvar,
        )

    def calculate_sharpe(
        self,
        returns: pd.Series,
        risk_free_rate: Optional[float] = None,
    ) -> float:
        """Calculate Sharpe ratio."""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        if HAS_EMPYRICAL:
            daily_rf = risk_free_rate / 252
            sharpe = empyrical.sharpe_ratio(returns, risk_free=daily_rf)
            return float(sharpe) if not np.isnan(sharpe) else 0
        else:
            annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
            annual_vol = returns.std() * np.sqrt(252)
            return (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

    def calculate_sortino(
        self,
        returns: pd.Series,
        required_return: Optional[float] = None,
    ) -> float:
        """Calculate Sortino ratio."""
        if required_return is None:
            required_return = self.risk_free_rate

        if HAS_EMPYRICAL:
            daily_req = required_return / 252
            sortino = empyrical.sortino_ratio(returns, required_return=daily_req)
            return float(sortino) if not np.isnan(sortino) else 0
        else:
            annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
            negative_returns = returns[returns < required_return / 252]
            downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            return (annual_return - required_return) / downside_vol if downside_vol > 0 else 0

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if HAS_EMPYRICAL:
            max_dd = empyrical.max_drawdown(returns)
            return float(abs(max_dd)) if not np.isnan(max_dd) else 0
        else:
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            return abs(drawdowns.min()) if len(drawdowns) > 0 else 0

    def calculate_rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 21,
    ) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        return (rolling_mean - self.risk_free_rate) / rolling_std

    def calculate_rolling_volatility(
        self,
        returns: pd.Series,
        window: int = 21,
    ) -> pd.Series:
        """Calculate rolling annualized volatility."""
        return returns.rolling(window).std() * np.sqrt(252)

    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk."""
        percentile = (1 - confidence) * 100
        return abs(np.percentile(returns, percentile))

    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        percentile = (1 - confidence) * 100
        var_threshold = np.percentile(returns, percentile)
        tail_returns = returns[returns <= var_threshold]
        return abs(tail_returns.mean()) if len(tail_returns) > 0 else self._calculate_var(returns, confidence)

    def compare_strategies(
        self,
        strategy_returns: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Args:
            strategy_returns: Dict mapping strategy name to returns

        Returns:
            DataFrame with metrics for each strategy
        """
        results = {}
        for name, returns in strategy_returns.items():
            metrics = self.calculate_metrics(returns)
            results[name] = metrics.to_dict()

        return pd.DataFrame(results).T
