"""
Portfolio state models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Position:
    """Represents a single portfolio position."""

    symbol: str
    shares: float
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    timestamp: Optional[datetime] = None

    @property
    def cost_basis(self) -> float:
        """Total cost basis for this position."""
        return self.shares * self.average_cost

    @property
    def weight(self) -> float:
        """Weight as a percentage of market value (requires portfolio context)."""
        return 0.0  # Calculated by PortfolioState

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "shares": self.shares,
            "average_cost": round(self.average_cost, 2),
            "current_price": round(self.current_price, 2),
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 4),
            "cost_basis": round(self.cost_basis, 2),
        }


@dataclass
class PortfolioState:
    """
    Current state of the portfolio.

    Loaded from Redis (robinhood:positions) and used for risk calculations.
    """

    positions: Dict[str, Position] = field(default_factory=dict)
    buying_power: float = 0.0
    cash: float = 0.0
    total_equity: float = 0.0
    market_value: float = 0.0
    last_updated: Optional[datetime] = None

    @property
    def total_positions_value(self) -> float:
        """Total market value of all positions."""
        return sum(p.market_value for p in self.positions.values())

    @property
    def position_count(self) -> int:
        """Number of open positions."""
        return len(self.positions)

    @property
    def cash_pct(self) -> float:
        """Cash as percentage of total equity."""
        if self.total_equity <= 0:
            return 1.0
        return self.cash / self.total_equity

    @property
    def symbols(self) -> List[str]:
        """List of symbols with open positions."""
        return list(self.positions.keys())

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in this symbol."""
        return symbol in self.positions

    def get_position_weight(self, symbol: str) -> float:
        """Get the weight of a position as percentage of total equity."""
        if self.total_equity <= 0:
            return 0.0
        position = self.positions.get(symbol)
        if not position:
            return 0.0
        return position.market_value / self.total_equity

    def get_total_exposure(self) -> float:
        """Get total exposure as percentage of equity."""
        if self.total_equity <= 0:
            return 0.0
        return self.total_positions_value / self.total_equity

    def get_weights(self) -> Dict[str, float]:
        """Get weights for all positions."""
        if self.total_equity <= 0:
            return {}
        return {
            symbol: pos.market_value / self.total_equity
            for symbol, pos in self.positions.items()
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "buying_power": round(self.buying_power, 2),
            "cash": round(self.cash, 2),
            "total_equity": round(self.total_equity, 2),
            "market_value": round(self.market_value, 2),
            "position_count": self.position_count,
            "cash_pct": round(self.cash_pct, 4),
            "total_exposure": round(self.get_total_exposure(), 4),
        }
