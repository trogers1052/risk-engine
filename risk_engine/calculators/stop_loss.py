"""
Stop loss calculator.

Calculates stop loss price and dollar amount based on position average cost
and exit strategy from rules.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from ..config import RiskSettings
from ..data.portfolio_state import PortfolioStateManager
from ..data.rules_client import RulesClient
from ..models.portfolio import Position

logger = logging.getLogger(__name__)


@dataclass
class StopLossCalculation:
    """Result of stop loss calculation."""

    symbol: str
    shares: float
    average_cost: float
    current_price: float
    stop_loss_pct: float
    stop_loss_price: float
    stop_loss_dollar_amount: float
    potential_loss: float
    potential_loss_pct: float
    distance_to_stop: float
    distance_to_stop_pct: float
    is_triggered: bool
    profit_target_pct: float
    profit_target_price: float

    def to_dict(self) -> dict:
        """Convert to dictionary for alerts/API."""
        return {
            "symbol": self.symbol,
            "shares": self.shares,
            "average_cost": round(self.average_cost, 2),
            "current_price": round(self.current_price, 2),
            "stop_loss_pct": round(self.stop_loss_pct * 100, 2),
            "stop_loss_price": round(self.stop_loss_price, 2),
            "stop_loss_dollar_amount": round(self.stop_loss_dollar_amount, 2),
            "potential_loss": round(self.potential_loss, 2),
            "potential_loss_pct": round(self.potential_loss_pct * 100, 2),
            "distance_to_stop": round(self.distance_to_stop, 2),
            "distance_to_stop_pct": round(self.distance_to_stop_pct * 100, 2),
            "is_triggered": self.is_triggered,
            "profit_target_pct": round(self.profit_target_pct * 100, 2),
            "profit_target_price": round(self.profit_target_price, 2),
        }

    def format_alert_message(self) -> str:
        """Format as human-readable alert message."""
        if self.is_triggered:
            status = "⚠️ STOP LOSS TRIGGERED"
        else:
            status = f"📊 Stop Loss Alert"

        return (
            f"{status}\n\n"
            f"Symbol: {self.symbol}\n"
            f"Shares: {self.shares:.0f}\n"
            f"Avg Cost: ${self.average_cost:.2f}\n"
            f"Current: ${self.current_price:.2f}\n\n"
            f"Stop Loss: ${self.stop_loss_price:.2f} ({self.stop_loss_pct*100:.1f}%)\n"
            f"Set Stop: ${self.stop_loss_dollar_amount:.2f}\n\n"
            f"Distance: ${self.distance_to_stop:.2f} ({self.distance_to_stop_pct*100:.1f}%)\n"
            f"Potential Loss: ${self.potential_loss:.2f}"
        )


class StopLossCalculator:
    """
    Calculates stop loss prices and amounts for positions.

    Uses:
    - Position average cost from PortfolioStateManager (Redis)
    - Exit strategy (stop_loss %) from RulesClient (Redis)
    """

    def __init__(
        self,
        settings: RiskSettings,
        portfolio_manager: PortfolioStateManager,
        rules_client: RulesClient,
    ):
        self.settings = settings
        self.portfolio_manager = portfolio_manager
        self.rules_client = rules_client

    def calculate(
        self,
        symbol: str,
        current_price: Optional[float] = None,
    ) -> Optional[StopLossCalculation]:
        """
        Calculate stop loss for a symbol.

        Args:
            symbol: Stock symbol
            current_price: Optional current price (uses position current_price if not provided)

        Returns:
            StopLossCalculation or None if no position exists
        """
        # Get position from portfolio
        portfolio = self.portfolio_manager.get_portfolio_state()
        position = portfolio.get_position(symbol)

        if not position:
            logger.debug(f"No position found for {symbol}")
            return None

        return self.calculate_for_position(position, current_price)

    def calculate_for_position(
        self,
        position: Position,
        current_price: Optional[float] = None,
    ) -> StopLossCalculation:
        """
        Calculate stop loss for a position.

        Args:
            position: Position object
            current_price: Optional current price override

        Returns:
            StopLossCalculation
        """
        symbol = position.symbol
        shares = position.shares
        average_cost = position.average_cost
        price = current_price if current_price is not None else position.current_price

        # Get exit strategy from rules
        exit_strategy = self.rules_client.get_exit_strategy(symbol)
        stop_loss_pct = exit_strategy.get("stop_loss", 0.05)
        profit_target_pct = exit_strategy.get("profit_target", 0.07)

        # Calculate stop loss price (below average cost)
        stop_loss_price = average_cost * (1 - stop_loss_pct)

        # Calculate profit target price
        profit_target_price = average_cost * (1 + profit_target_pct)

        # Total dollar amount to set as stop loss order
        # This is what you'd enter as the stop loss value in Robinhood
        stop_loss_dollar_amount = stop_loss_price * shares

        # Potential loss if stop loss is hit
        potential_loss = (average_cost - stop_loss_price) * shares

        # Potential loss as percentage of position value
        position_value = average_cost * shares
        potential_loss_pct = (
            potential_loss / position_value if position_value > 0 else 0
        )

        # Distance from current price to stop loss
        distance_to_stop = price - stop_loss_price
        distance_to_stop_pct = (
            distance_to_stop / price if price > 0 else 0
        )

        # Check if stop loss is already triggered
        is_triggered = price <= stop_loss_price

        return StopLossCalculation(
            symbol=symbol,
            shares=shares,
            average_cost=average_cost,
            current_price=price,
            stop_loss_pct=stop_loss_pct,
            stop_loss_price=stop_loss_price,
            stop_loss_dollar_amount=stop_loss_dollar_amount,
            potential_loss=potential_loss,
            potential_loss_pct=potential_loss_pct,
            distance_to_stop=distance_to_stop,
            distance_to_stop_pct=distance_to_stop_pct,
            is_triggered=is_triggered,
            profit_target_pct=profit_target_pct,
            profit_target_price=profit_target_price,
        )

    def calculate_all(
        self,
        current_prices: Optional[dict] = None,
    ) -> list[StopLossCalculation]:
        """
        Calculate stop losses for all positions.

        Args:
            current_prices: Optional dict of symbol -> current price

        Returns:
            List of StopLossCalculation for all positions
        """
        portfolio = self.portfolio_manager.get_portfolio_state()
        results = []

        for symbol, position in portfolio.positions.items():
            price = (
                current_prices.get(symbol) if current_prices else None
            )
            calc = self.calculate_for_position(position, price)
            results.append(calc)

        return results

    def get_triggered_stops(
        self,
        current_prices: Optional[dict] = None,
    ) -> list[StopLossCalculation]:
        """
        Get all positions where stop loss has been triggered.

        Args:
            current_prices: Optional dict of symbol -> current price

        Returns:
            List of triggered StopLossCalculations
        """
        all_calcs = self.calculate_all(current_prices)
        return [calc for calc in all_calcs if calc.is_triggered]

    def calculate_from_entry(
        self,
        symbol: str,
        entry_price: float,
        shares: float,
        current_price: Optional[float] = None,
    ) -> StopLossCalculation:
        """
        Calculate stop loss from entry parameters (for new positions).

        Use this when you want to calculate stop loss before position
        is reflected in the portfolio.

        Args:
            symbol: Stock symbol
            entry_price: Entry/average price
            shares: Number of shares
            current_price: Current price (defaults to entry_price)

        Returns:
            StopLossCalculation
        """
        # Create a temporary position object
        price = current_price if current_price is not None else entry_price
        position = Position(
            symbol=symbol,
            shares=shares,
            average_cost=entry_price,
            current_price=price,
            market_value=price * shares,
            unrealized_pnl=(price - entry_price) * shares,
            unrealized_pnl_pct=(price - entry_price) / entry_price if entry_price > 0 else 0,
        )

        return self.calculate_for_position(position, price)
