"""
Base risk check interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..config import RiskSettings
from ..models.portfolio import PortfolioState
from ..models.risk_result import RiskCheckResult


@dataclass
class RiskCheckContext:
    """
    Context passed to risk checks.

    Contains all information needed for risk evaluation.
    """

    # Symbol being evaluated
    symbol: str

    # Signal information
    signal_type: str  # "BUY", "SELL", "WATCH"
    confidence: float

    # Current indicators from decision-engine
    indicators: Dict[str, float] = field(default_factory=dict)

    # Current portfolio state
    portfolio: Optional[PortfolioState] = None

    # Current price (from indicators or provided)
    current_price: Optional[float] = None

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_indicator(self, name: str, default: Optional[float] = None) -> Optional[float]:
        """Get an indicator value by name."""
        return self.indicators.get(name, default)

    @property
    def atr(self) -> Optional[float]:
        """Get ATR from indicators."""
        return self.get_indicator("atr_14") or self.get_indicator("atr")

    @property
    def atr_pct(self) -> Optional[float]:
        """Get ATR as percentage of price."""
        return self.get_indicator("atr_pct")

    @property
    def sma_200(self) -> Optional[float]:
        """Get 200-day SMA."""
        return self.get_indicator("sma_200")

    @property
    def volume(self) -> Optional[float]:
        """Get current volume."""
        return self.get_indicator("volume")

    @property
    def avg_volume(self) -> Optional[float]:
        """Get average volume."""
        return self.get_indicator("avg_volume_20") or self.get_indicator("avg_volume")

    @property
    def rsi(self) -> Optional[float]:
        """Get RSI."""
        return self.get_indicator("rsi_14") or self.get_indicator("rsi")

    @property
    def price(self) -> Optional[float]:
        """Get current price."""
        if self.current_price:
            return self.current_price
        return self.get_indicator("close") or self.get_indicator("price")


class RiskCheck(ABC):
    """
    Abstract base class for risk checks.

    Each risk check evaluates a specific aspect of trade risk
    and returns a RiskCheckResult.
    """

    def __init__(self, settings: RiskSettings):
        self.settings = settings

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this risk check."""
        pass

    @property
    def description(self) -> str:
        """Description of what this check evaluates."""
        return ""

    @abstractmethod
    def check(self, context: RiskCheckContext) -> RiskCheckResult:
        """
        Evaluate risk for the given context.

        Args:
            context: Risk check context with symbol, indicators, portfolio

        Returns:
            RiskCheckResult with pass/fail and details
        """
        pass

    def can_check(self, context: RiskCheckContext) -> bool:
        """
        Check if this risk check can be evaluated.

        Override to specify required indicators or conditions.

        Args:
            context: Risk check context

        Returns:
            True if check can be performed
        """
        return True

    def _reject(
        self,
        reason: str,
        risk_score: float = 1.0,
        context: Optional[RiskCheckContext] = None,
        **kwargs,
    ) -> RiskCheckResult:
        """Helper to create a rejection result."""
        return RiskCheckResult.reject(
            reason=reason,
            risk_score=risk_score,
            check_name=self.name,
            symbol=context.symbol if context else None,
            **kwargs,
        )

    def _approve(
        self,
        recommended_shares: int,
        max_shares: int,
        recommended_dollar_amount: float,
        risk_score: float,
        context: Optional[RiskCheckContext] = None,
        reason: str = "Check passed",
        **kwargs,
    ) -> RiskCheckResult:
        """Helper to create an approval result."""
        return RiskCheckResult.approve(
            recommended_shares=recommended_shares,
            max_shares=max_shares,
            recommended_dollar_amount=recommended_dollar_amount,
            risk_score=risk_score,
            reason=reason,
            check_name=self.name,
            symbol=context.symbol if context else None,
            **kwargs,
        )
