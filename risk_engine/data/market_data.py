"""
Market data loader - loads historical price data from TimescaleDB.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from ..config import RiskSettings

logger = logging.getLogger(__name__)


class MarketDataLoader:
    """
    Loads historical market data from TimescaleDB.

    Used for VaR/CVaR calculations and correlation analysis.
    """

    def __init__(self, settings: RiskSettings):
        self.settings = settings
        self._conn = None

    def connect(self) -> bool:
        """Connect to TimescaleDB."""
        try:
            self._conn = psycopg2.connect(
                host=self.settings.timescale_host,
                port=self.settings.timescale_port,
                dbname=self.settings.timescale_db,
                user=self.settings.timescale_user,
                password=self.settings.timescale_password,
            )
            logger.info(
                f"Connected to TimescaleDB at {self.settings.timescale_host}:"
                f"{self.settings.timescale_port}"
            )
            return True
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            return False

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_daily_returns(
        self,
        symbol: str,
        lookback_days: int = 252,
    ) -> Optional[pd.Series]:
        """
        Get daily returns for a symbol.

        Args:
            symbol: Stock symbol
            lookback_days: Number of trading days to look back

        Returns:
            Series of daily returns indexed by date
        """
        prices = self.get_daily_prices(symbol, lookback_days)
        if prices is None or len(prices) < 2:
            return None

        returns = prices.pct_change().dropna()
        return returns

    def get_daily_prices(
        self,
        symbol: str,
        lookback_days: int = 252,
    ) -> Optional[pd.Series]:
        """
        Get daily closing prices for a symbol.

        Args:
            symbol: Stock symbol
            lookback_days: Number of calendar days to look back

        Returns:
            Series of daily prices indexed by date
        """
        if not self._conn:
            logger.warning("Database not connected")
            return None

        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days * 2)

            query = """
                SELECT
                    time_bucket('1 day', time) AS date,
                    last(close, time) AS close
                FROM ohlcv_1min
                WHERE symbol = %s
                  AND time >= %s
                  AND time <= %s
                GROUP BY date
                ORDER BY date
                LIMIT %s
            """

            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (symbol, start_date, end_date, lookback_days))
                rows = cur.fetchall()

            if not rows:
                logger.debug(f"No price data found for {symbol}")
                return None

            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            return df["close"]

        except psycopg2.Error as e:
            logger.error(f"Database error fetching prices for {symbol}: {e}")
            return None

    def get_multiple_returns(
        self,
        symbols: List[str],
        lookback_days: int = 252,
    ) -> Optional[pd.DataFrame]:
        """
        Get daily returns for multiple symbols.

        Args:
            symbols: List of stock symbols
            lookback_days: Number of trading days

        Returns:
            DataFrame with columns for each symbol's returns
        """
        if not symbols:
            return None

        returns_dict = {}
        for symbol in symbols:
            returns = self.get_daily_returns(symbol, lookback_days)
            if returns is not None and len(returns) > 0:
                returns_dict[symbol] = returns

        if not returns_dict:
            return None

        df = pd.DataFrame(returns_dict)
        # Forward fill then back fill to align dates
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    def get_correlation_matrix(
        self,
        symbols: List[str],
        lookback_days: int = 60,
    ) -> Optional[pd.DataFrame]:
        """
        Calculate correlation matrix for a set of symbols.

        Args:
            symbols: List of stock symbols
            lookback_days: Lookback period for correlation

        Returns:
            DataFrame correlation matrix
        """
        returns_df = self.get_multiple_returns(symbols, lookback_days)
        if returns_df is None or len(returns_df) < 10:
            return None

        return returns_df.corr()

    def get_correlation_with_portfolio(
        self,
        symbol: str,
        portfolio_symbols: List[str],
        lookback_days: int = 60,
    ) -> Dict[str, float]:
        """
        Calculate correlation of a symbol with existing portfolio positions.

        Args:
            symbol: Symbol to check
            portfolio_symbols: Existing portfolio symbols
            lookback_days: Lookback period

        Returns:
            Dict mapping portfolio symbol to correlation
        """
        all_symbols = [symbol] + portfolio_symbols
        corr_matrix = self.get_correlation_matrix(all_symbols, lookback_days)

        if corr_matrix is None or symbol not in corr_matrix.columns:
            return {}

        correlations = {}
        for port_symbol in portfolio_symbols:
            if port_symbol in corr_matrix.columns:
                correlations[port_symbol] = corr_matrix.loc[symbol, port_symbol]

        return correlations

    def get_max_correlation_with_portfolio(
        self,
        symbol: str,
        portfolio_symbols: List[str],
        lookback_days: int = 60,
    ) -> Tuple[Optional[str], float]:
        """
        Find the maximum correlation with any portfolio position.

        Args:
            symbol: Symbol to check
            portfolio_symbols: Existing portfolio symbols
            lookback_days: Lookback period

        Returns:
            Tuple of (most correlated symbol, correlation value)
        """
        correlations = self.get_correlation_with_portfolio(
            symbol, portfolio_symbols, lookback_days
        )

        if not correlations:
            return None, 0.0

        max_symbol = max(correlations.keys(), key=lambda k: abs(correlations[k]))
        return max_symbol, correlations[max_symbol]

    def get_average_volume(
        self,
        symbol: str,
        lookback_days: int = 20,
    ) -> Optional[float]:
        """
        Get average daily volume for a symbol.

        Args:
            symbol: Stock symbol
            lookback_days: Lookback period

        Returns:
            Average daily volume
        """
        if not self._conn:
            logger.warning("Database not connected")
            return None

        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days * 2)

            query = """
                SELECT
                    time_bucket('1 day', time) AS date,
                    sum(volume) AS daily_volume
                FROM ohlcv_1min
                WHERE symbol = %s
                  AND time >= %s
                  AND time <= %s
                GROUP BY date
                ORDER BY date DESC
                LIMIT %s
            """

            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (symbol, start_date, end_date, lookback_days))
                rows = cur.fetchall()

            if not rows:
                return None

            volumes = [row["daily_volume"] for row in rows if row["daily_volume"]]
            if not volumes:
                return None

            return np.mean(volumes)

        except psycopg2.Error as e:
            logger.error(f"Database error fetching volume for {symbol}: {e}")
            return None

    def get_portfolio_returns(
        self,
        weights: Dict[str, float],
        lookback_days: int = 252,
    ) -> Optional[pd.Series]:
        """
        Calculate weighted portfolio returns.

        Args:
            weights: Dict mapping symbol to weight
            lookback_days: Lookback period

        Returns:
            Series of portfolio returns
        """
        symbols = list(weights.keys())
        returns_df = self.get_multiple_returns(symbols, lookback_days)

        if returns_df is None:
            return None

        # Create weights array aligned with columns
        weight_array = np.array(
            [weights.get(col, 0) for col in returns_df.columns]
        )

        # Normalize weights
        weight_sum = weight_array.sum()
        if weight_sum > 0:
            weight_array = weight_array / weight_sum

        # Calculate weighted returns
        portfolio_returns = (returns_df * weight_array).sum(axis=1)
        return portfolio_returns
