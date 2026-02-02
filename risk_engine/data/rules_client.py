"""
Rules client for reading rules from Redis cache.

Reads rules published by decision-engine for stop loss calculations.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional

import redis

from ..config import RiskSettings

logger = logging.getLogger(__name__)

# Redis keys (must match decision-engine's rules_cache.py)
RULES_CONFIG_KEY = "trading:rules:config"
RULES_UPDATED_KEY = "trading:rules:updated_at"
SYMBOL_RULES_PREFIX = "trading:rules:symbol:"
EXIT_STRATEGY_KEY = "trading:rules:exit_strategy"


class RulesClient:
    """
    Client for reading rules from Redis cache.

    Used by risk-engine to get exit strategy (stop loss, profit target)
    for calculating stop loss alerts.
    """

    def __init__(self, settings: RiskSettings):
        self.settings = settings
        self._redis: Optional[redis.Redis] = None
        self._config_cache: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 60  # Refresh from Redis every 60 seconds

    def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self._redis = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                decode_responses=True,
            )
            self._redis.ping()
            logger.info(
                f"Rules client connected to Redis at "
                f"{self.settings.redis_host}:{self.settings.redis_port}"
            )
            return True
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis for rules: {e}")
            return False

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            self._redis.close()
            self._redis = None

    def get_exit_strategy(self, symbol: str) -> Dict[str, float]:
        """
        Get exit strategy for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with profit_target and stop_loss percentages
        """
        default = {"profit_target": 0.07, "stop_loss": 0.05}

        if not self._redis:
            logger.debug("Redis not connected, using default exit strategy")
            return default

        try:
            # Check symbol-specific first
            symbol_data = self._redis.get(f"{SYMBOL_RULES_PREFIX}{symbol}")
            if symbol_data:
                override = json.loads(symbol_data)
                if "exit_strategy" in override:
                    logger.debug(
                        f"Using symbol-specific exit strategy for {symbol}"
                    )
                    return override["exit_strategy"]

            # Fall back to default
            default_data = self._redis.get(EXIT_STRATEGY_KEY)
            if default_data:
                return json.loads(default_data)

        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error getting exit strategy: {e}")

        return default

    def get_stop_loss_pct(self, symbol: str) -> float:
        """
        Get stop loss percentage for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Stop loss as a decimal (e.g., 0.05 for 5%)
        """
        exit_strategy = self.get_exit_strategy(symbol)
        return exit_strategy.get("stop_loss", 0.05)

    def get_profit_target_pct(self, symbol: str) -> float:
        """
        Get profit target percentage for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Profit target as a decimal (e.g., 0.07 for 7%)
        """
        exit_strategy = self.get_exit_strategy(symbol)
        return exit_strategy.get("profit_target", 0.07)

    def get_config(self, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get the full rules configuration.

        Args:
            force_refresh: Force reload from Redis

        Returns:
            Full rules config dict or None
        """
        if not self._redis:
            return self._config_cache

        # Check cache
        now = datetime.utcnow()
        if (
            not force_refresh
            and self._config_cache
            and self._cache_time
            and (now - self._cache_time).total_seconds() < self._cache_ttl_seconds
        ):
            return self._config_cache

        try:
            data = self._redis.get(RULES_CONFIG_KEY)
            if data:
                self._config_cache = json.loads(data)
                self._cache_time = now
                return self._config_cache
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error loading rules config: {e}")

        return self._config_cache

    def get_last_updated(self) -> Optional[datetime]:
        """Get when rules were last updated."""
        if not self._redis:
            return None

        try:
            data = self._redis.get(RULES_UPDATED_KEY)
            if data:
                return datetime.fromisoformat(data)
        except (redis.RedisError, ValueError):
            pass

        return None
