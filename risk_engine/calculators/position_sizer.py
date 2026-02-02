"""
Position sizing calculator.

Implements ATR-based and Kelly criterion position sizing methods.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import RiskSettings

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""

    shares: int
    dollar_amount: float
    position_pct: float
    method: str
    risk_per_share: float
    max_shares: int
    max_dollar_amount: float
    notes: str = ""


class PositionSizer:
    """
    Calculate position sizes using various methods.

    Methods:
    - ATR-based: Position = (Portfolio * RiskPct) / (ATR * Multiplier)
    - Kelly: f* = (mean - rf) / variance, adjusted by fraction
    - Fixed percentage: Simple percentage of portfolio
    """

    def __init__(self, settings: RiskSettings):
        self.settings = settings

    def calculate(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        buying_power: float,
        atr: Optional[float] = None,
        returns: Optional[pd.Series] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> PositionSizeResult:
        """
        Calculate recommended position size.

        Args:
            symbol: Stock symbol
            price: Current stock price
            portfolio_value: Total portfolio value
            buying_power: Available buying power
            atr: Average True Range (for ATR method)
            returns: Historical returns series (for Kelly method)
            win_rate: Historical win rate (for Kelly method)
            avg_win: Average winning trade (for Kelly method)
            avg_loss: Average losing trade (for Kelly method)

        Returns:
            PositionSizeResult with sizing details
        """
        method = self.settings.position_sizing.method
        max_position_pct = self.settings.get_max_position_pct(symbol)
        max_dollar_amount = portfolio_value * max_position_pct
        max_shares = int(max_dollar_amount / price) if price > 0 else 0

        if method == "kelly":
            result = self._kelly_sizing(
                symbol, price, portfolio_value, returns, win_rate, avg_win, avg_loss
            )
        elif method == "fixed":
            result = self._fixed_sizing(symbol, price, portfolio_value)
        else:  # Default to ATR
            result = self._atr_sizing(symbol, price, portfolio_value, atr)

        if result is None:
            # Fallback to conservative fixed sizing
            result = self._fixed_sizing(symbol, price, portfolio_value)
            result.notes = f"Fallback from {method} to fixed sizing"

        # Apply limits
        result.max_shares = max_shares
        result.max_dollar_amount = max_dollar_amount

        if result.dollar_amount > max_dollar_amount:
            result.shares = max_shares
            result.dollar_amount = max_shares * price
            result.position_pct = max_position_pct
            result.notes += f" Capped at max position {max_position_pct:.1%}"

        if result.dollar_amount > buying_power:
            affordable_shares = int(buying_power / price) if price > 0 else 0
            result.shares = affordable_shares
            result.dollar_amount = affordable_shares * price
            result.position_pct = result.dollar_amount / portfolio_value
            result.notes += f" Limited by buying power ${buying_power:.2f}"

        return result

    def _atr_sizing(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        atr: Optional[float],
    ) -> Optional[PositionSizeResult]:
        """
        Calculate position size using ATR-based method.

        Formula: Shares = (Portfolio * RiskPct) / (ATR * Multiplier)
        """
        if atr is None or atr <= 0:
            return None

        risk_pct = self.settings.position_sizing.risk_per_trade_pct
        multiplier = self.settings.position_sizing.atr_multiplier

        # Risk amount per trade
        risk_amount = portfolio_value * risk_pct

        # Stop distance (risk per share)
        stop_distance = atr * multiplier

        # Number of shares
        shares = int(risk_amount / stop_distance)
        dollar_amount = shares * price
        position_pct = dollar_amount / portfolio_value if portfolio_value > 0 else 0

        logger.debug(
            f"ATR sizing {symbol}: "
            f"ATR={atr:.4f}, stop={stop_distance:.4f}, "
            f"risk_amount=${risk_amount:.2f}, shares={shares}"
        )

        return PositionSizeResult(
            shares=shares,
            dollar_amount=dollar_amount,
            position_pct=position_pct,
            method="atr",
            risk_per_share=stop_distance,
            max_shares=0,
            max_dollar_amount=0,
            notes=f"ATR={atr:.4f}, multiplier={multiplier}",
        )

    def _kelly_sizing(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        returns: Optional[pd.Series] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> Optional[PositionSizeResult]:
        """
        Calculate position size using Kelly criterion.

        Classical Kelly: f* = p - (1-p)/b
        where p = win rate, b = win/loss ratio

        Alternative (returns-based): f* = mean / variance
        """
        kelly_fraction = self.settings.position_sizing.kelly_fraction

        if returns is not None and len(returns) >= 30:
            # Use returns-based Kelly
            mean_return = returns.mean()
            variance = returns.var()

            if variance <= 0:
                return None

            # Kelly fraction
            kelly_f = mean_return / variance

            # Apply fractional Kelly
            adjusted_f = kelly_f * kelly_fraction

            # Cap at max position
            max_pct = self.settings.get_max_position_pct(symbol)
            position_pct = min(max(adjusted_f, 0), max_pct)

        elif win_rate is not None and avg_win is not None and avg_loss is not None:
            # Use win/loss ratio Kelly
            if avg_loss <= 0:
                return None

            b = avg_win / abs(avg_loss)  # Win/loss ratio
            p = win_rate

            # Kelly formula
            kelly_f = p - (1 - p) / b

            # Apply fractional Kelly
            adjusted_f = kelly_f * kelly_fraction

            # Cap at max position
            max_pct = self.settings.get_max_position_pct(symbol)
            position_pct = min(max(adjusted_f, 0), max_pct)
        else:
            return None

        dollar_amount = portfolio_value * position_pct
        shares = int(dollar_amount / price) if price > 0 else 0

        logger.debug(
            f"Kelly sizing {symbol}: "
            f"kelly_f={kelly_f:.4f}, fraction={kelly_fraction}, "
            f"position_pct={position_pct:.4f}, shares={shares}"
        )

        return PositionSizeResult(
            shares=shares,
            dollar_amount=dollar_amount,
            position_pct=position_pct,
            method="kelly",
            risk_per_share=0,
            max_shares=0,
            max_dollar_amount=0,
            notes=f"Kelly f*={kelly_f:.4f}, fraction={kelly_fraction}",
        )

    def _fixed_sizing(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
    ) -> PositionSizeResult:
        """
        Calculate position size using fixed percentage.

        Uses risk_per_trade_pct as the position size percentage.
        """
        # Use a conservative fixed percentage
        position_pct = self.settings.position_sizing.risk_per_trade_pct * 5
        max_pct = self.settings.get_max_position_pct(symbol)
        position_pct = min(position_pct, max_pct)

        dollar_amount = portfolio_value * position_pct
        shares = int(dollar_amount / price) if price > 0 else 0

        return PositionSizeResult(
            shares=shares,
            dollar_amount=dollar_amount,
            position_pct=position_pct,
            method="fixed",
            risk_per_share=0,
            max_shares=0,
            max_dollar_amount=0,
            notes=f"Fixed {position_pct:.1%} allocation",
        )

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        multiplier: Optional[float] = None,
    ) -> float:
        """
        Calculate stop loss price based on ATR.

        Args:
            entry_price: Entry price
            atr: Average True Range
            multiplier: ATR multiplier (default from config)

        Returns:
            Stop loss price
        """
        if multiplier is None:
            multiplier = self.settings.position_sizing.atr_multiplier

        stop_distance = atr * multiplier
        return entry_price - stop_distance

    def calculate_position_for_risk(
        self,
        entry_price: float,
        stop_price: float,
        risk_amount: float,
    ) -> int:
        """
        Calculate position size for a given risk amount and stop.

        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            risk_amount: Maximum $ risk

        Returns:
            Number of shares
        """
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0

        shares = int(risk_amount / risk_per_share)
        return max(0, shares)
