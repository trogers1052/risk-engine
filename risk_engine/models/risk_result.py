"""
Risk check result models.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class RiskLevel(Enum):
    """Risk level classification."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Determine risk level from a 0.0-1.0 score."""
        if score < 0.25:
            return cls.LOW
        elif score < 0.50:
            return cls.MEDIUM
        elif score < 0.75:
            return cls.HIGH
        else:
            return cls.CRITICAL


@dataclass
class RiskCheckResult:
    """
    Result of a risk check evaluation.

    This is the primary output of the risk engine, containing
    the decision, position sizing recommendation, and detailed metrics.
    """

    # Core decision
    passes: bool
    reason: str

    # Risk assessment
    risk_score: float  # 0.0-1.0 overall risk score
    risk_level: RiskLevel

    # Position sizing
    recommended_shares: int
    max_shares: int
    recommended_dollar_amount: float

    # Detailed metrics
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # Optional metadata
    check_name: Optional[str] = None
    symbol: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "passes": self.passes,
            "reason": self.reason,
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level.value,
            "recommended_shares": self.recommended_shares,
            "max_shares": self.max_shares,
            "recommended_dollar_amount": round(self.recommended_dollar_amount, 2),
            "risk_metrics": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.risk_metrics.items()
            },
            "warnings": self.warnings,
            "check_name": self.check_name,
            "symbol": self.symbol,
        }

    @classmethod
    def reject(
        cls,
        reason: str,
        risk_score: float = 1.0,
        check_name: Optional[str] = None,
        symbol: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        risk_metrics: Optional[Dict[str, float]] = None,
    ) -> "RiskCheckResult":
        """Create a rejection result."""
        return cls(
            passes=False,
            reason=reason,
            risk_score=risk_score,
            risk_level=RiskLevel.from_score(risk_score),
            recommended_shares=0,
            max_shares=0,
            recommended_dollar_amount=0.0,
            check_name=check_name,
            symbol=symbol,
            warnings=warnings or [],
            risk_metrics=risk_metrics or {},
        )

    @classmethod
    def approve(
        cls,
        recommended_shares: int,
        max_shares: int,
        recommended_dollar_amount: float,
        risk_score: float,
        reason: str = "All risk checks passed",
        check_name: Optional[str] = None,
        symbol: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        risk_metrics: Optional[Dict[str, float]] = None,
    ) -> "RiskCheckResult":
        """Create an approval result."""
        return cls(
            passes=True,
            reason=reason,
            risk_score=risk_score,
            risk_level=RiskLevel.from_score(risk_score),
            recommended_shares=recommended_shares,
            max_shares=max_shares,
            recommended_dollar_amount=recommended_dollar_amount,
            check_name=check_name,
            symbol=symbol,
            warnings=warnings or [],
            risk_metrics=risk_metrics or {},
        )

    def merge_warnings(self, other: "RiskCheckResult") -> None:
        """Merge warnings from another result."""
        self.warnings.extend(other.warnings)

    def merge_metrics(self, other: "RiskCheckResult") -> None:
        """Merge risk metrics from another result."""
        self.risk_metrics.update(other.risk_metrics)
