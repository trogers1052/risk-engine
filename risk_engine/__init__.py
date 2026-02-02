"""
Risk Engine - Portfolio risk management and position sizing.

A Python module for evaluating trade risks, calculating position sizes,
and managing portfolio risk before buy signals are published.
"""

from .models.risk_result import RiskCheckResult, RiskLevel
from .models.portfolio import Position, PortfolioState
from .checker import RiskChecker
from .adapter import RiskAdapter

__version__ = "0.1.0"
__all__ = [
    "RiskCheckResult",
    "RiskLevel",
    "Position",
    "PortfolioState",
    "RiskChecker",
    "RiskAdapter",
]
