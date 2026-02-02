"""
Portfolio state manager - loads current positions from Redis.
"""

import json
import logging
from datetime import datetime
from typing import Optional

import redis

from ..config import RiskSettings
from ..models.portfolio import Position, PortfolioState

logger = logging.getLogger(__name__)


class PortfolioStateManager:
    """
    Manages portfolio state loaded from Redis.

    Reads from the robinhood:positions key which is populated
    by the robinhood-sync service.
    """

    def __init__(self, settings: RiskSettings):
        self.settings = settings
        self._redis: Optional[redis.Redis] = None
        self._cache: Optional[PortfolioState] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 5  # Refresh every 5 seconds

    def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self._redis = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                decode_responses=True,
            )
            # Test connection
            self._redis.ping()
            logger.info(
                f"Connected to Redis at {self.settings.redis_host}:"
                f"{self.settings.redis_port}"
            )
            return True
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            self._redis.close()
            self._redis = None

    def get_portfolio_state(self, force_refresh: bool = False) -> PortfolioState:
        """
        Get current portfolio state.

        Args:
            force_refresh: Force reload from Redis even if cached

        Returns:
            Current portfolio state
        """
        now = datetime.utcnow()

        # Check cache
        if not force_refresh and self._cache and self._cache_time:
            elapsed = (now - self._cache_time).total_seconds()
            if elapsed < self._cache_ttl_seconds:
                return self._cache

        # Load from Redis
        state = self._load_from_redis()
        self._cache = state
        self._cache_time = now
        return state

    def _load_from_redis(self) -> PortfolioState:
        """Load portfolio state from Redis."""
        if not self._redis:
            logger.warning("Redis not connected, returning empty portfolio")
            return PortfolioState()

        try:
            raw_data = self._redis.get(self.settings.redis_positions_key)
            if not raw_data:
                logger.debug("No portfolio data in Redis")
                return PortfolioState()

            data = json.loads(raw_data)
            return self._parse_portfolio_data(data)

        except redis.RedisError as e:
            logger.error(f"Redis error loading portfolio: {e}")
            return self._cache or PortfolioState()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in portfolio data: {e}")
            return self._cache or PortfolioState()

    def _parse_portfolio_data(self, data: dict) -> PortfolioState:
        """Parse raw Redis data into PortfolioState."""
        positions = {}

        # Parse positions array
        for pos_data in data.get("positions", []):
            try:
                symbol = pos_data.get("symbol")
                if not symbol:
                    continue

                position = Position(
                    symbol=symbol,
                    shares=float(pos_data.get("quantity", 0)),
                    average_cost=float(pos_data.get("average_buy_price", 0)),
                    current_price=float(pos_data.get("current_price", 0)),
                    market_value=float(pos_data.get("market_value", 0)),
                    unrealized_pnl=float(pos_data.get("unrealized_pnl", 0)),
                    unrealized_pnl_pct=float(
                        pos_data.get("unrealized_pnl_pct", 0)
                    ),
                )
                positions[symbol] = position
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing position: {e}")
                continue

        # Parse account-level data
        account = data.get("account", {})

        # Handle timestamp
        last_updated = None
        ts_str = data.get("timestamp") or data.get("last_updated")
        if ts_str:
            try:
                last_updated = datetime.fromisoformat(
                    ts_str.replace("Z", "+00:00")
                )
            except ValueError:
                pass

        return PortfolioState(
            positions=positions,
            buying_power=float(account.get("buying_power", 0)),
            cash=float(account.get("cash", 0)),
            total_equity=float(account.get("total_equity", 0)),
            market_value=float(account.get("market_value", 0)),
            last_updated=last_updated,
        )

    def get_position_weight(self, symbol: str) -> float:
        """Get position weight for a symbol."""
        state = self.get_portfolio_state()
        return state.get_position_weight(symbol)

    def get_buying_power(self) -> float:
        """Get current buying power."""
        state = self.get_portfolio_state()
        return state.buying_power

    def get_total_equity(self) -> float:
        """Get total portfolio equity."""
        state = self.get_portfolio_state()
        return state.total_equity

    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in a symbol."""
        state = self.get_portfolio_state()
        return state.has_position(symbol)
