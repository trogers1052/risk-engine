"""
Portfolio state manager - loads current positions from Redis.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
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
                password=self.settings.redis_password or None,
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
        """Load portfolio state from Redis.

        robinhood-sync stores positions as a Redis hash (one key per symbol)
        at ``robinhood:positions`` and account data as a JSON string at
        ``robinhood:buying_power``.  We read both and combine them into a
        single PortfolioState.
        """
        if not self._redis:
            logger.warning("Redis not connected, returning empty portfolio")
            return PortfolioState()

        try:
            # Read positions hash — each value is a JSON-encoded position
            positions_hash = self._redis.hgetall(
                self.settings.redis_positions_key
            )

            # Read account-level data (buying power, cash, equity)
            account_raw = self._redis.get("robinhood:buying_power")
            account = json.loads(account_raw) if account_raw else {}

            # Combine into the format _parse_portfolio_data expects
            positions_list = []
            for _symbol, pos_json in positions_hash.items():
                try:
                    positions_list.append(json.loads(pos_json))
                except json.JSONDecodeError:
                    logger.warning(f"Bad JSON for position {_symbol}")

            data = {"positions": positions_list, "account": account}
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

        # Parse positions array.
        # Robinhood-sync stores: quantity, average_buy_price, equity,
        # equity_change, percent_change.  Map to Position fields.
        for pos_data in data.get("positions", []):
            try:
                symbol = pos_data.get("symbol")
                if not symbol:
                    continue

                shares = float(pos_data.get("quantity", 0))
                avg_cost = float(pos_data.get("average_buy_price", 0))
                market_value = float(pos_data.get("equity", 0))
                current_price = (
                    market_value / shares if shares > 0 else avg_cost
                )
                unrealized_pnl = float(
                    pos_data.get("equity_change", 0)
                )
                # percent_change is stored as e.g. "-4.97" meaning -4.97%
                unrealized_pnl_pct = (
                    float(pos_data.get("percent_change", 0)) / 100.0
                )

                position = Position(
                    symbol=symbol,
                    shares=shares,
                    average_cost=avg_cost,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                )
                positions[symbol] = position
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing position {pos_data.get('symbol', '?')}: {e}")
                continue

        # Parse account-level data
        account = data.get("account", {})

        # Handle timestamp (from account data)
        last_updated = None
        ts_str = account.get("updated_at") or data.get("timestamp")
        if ts_str:
            try:
                last_updated = datetime.fromisoformat(
                    ts_str.replace("Z", "+00:00")
                )
            except ValueError:
                pass

        total_equity = float(account.get("total_equity", 0))
        cash = float(account.get("cash", 0))
        # market_value = sum of position equities (or equity - cash)
        market_value = float(account.get("market_value", 0))
        if market_value == 0 and positions:
            market_value = sum(p.market_value for p in positions.values())

        return PortfolioState(
            positions=positions,
            buying_power=float(account.get("buying_power", 0)),
            cash=cash,
            total_equity=total_equity,
            market_value=market_value,
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

    # ------------------------------------------------------------------
    # Macro signals (VIX, HY spread) from context-service
    # ------------------------------------------------------------------

    def get_macro_signals(self) -> dict:
        """Read VIX and HY spread from market:context (published by context-service).

        Returns {"vix": float, "hy_spread": float} or empty dict on failure.
        """
        if not self._redis:
            return {}
        try:
            raw = self._redis.get("market:context")
            if not raw:
                return {}
            data = json.loads(raw)
            macro = data.get("macro_signals", {})
            result = {}
            if macro.get("available"):
                if "vix" in macro:
                    result["vix"] = float(macro["vix"])
                if "hy_spread" in macro:
                    result["hy_spread"] = float(macro["hy_spread"])
            return result
        except (redis.RedisError, json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to read macro signals from Redis: {exc}")
            return {}

    def get_regime(self) -> str:
        """Read market regime from market:context (published by context-service).

        Returns regime string (BULL/BEAR/SIDEWAYS/UNKNOWN).
        Defaults to UNKNOWN on any failure (fail-open).
        """
        if not self._redis:
            return "UNKNOWN"
        try:
            raw = self._redis.get("market:context")
            if not raw:
                return "UNKNOWN"
            data = json.loads(raw)
            return str(data.get("regime", "UNKNOWN")).upper()
        except (redis.RedisError, json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to read regime from Redis: {exc}")
            return "UNKNOWN"

    # ------------------------------------------------------------------
    # Peak equity tracking for drawdown circuit breaker
    # ------------------------------------------------------------------

    PEAK_EQUITY_KEY = "risk:peak_equity"
    _PEAK_EQUITY_FILE = os.environ.get(
        "RISK_PEAK_EQUITY_FILE",
        str(Path.home() / ".risk-engine" / "peak_equity"),
    )

    def get_peak_equity(self, current_equity: float) -> float:
        """Get peak equity, updating Redis + file backup if current equity is a new high.

        Resolution order:
        1. Redis (fast, primary)
        2. Local file backup (survives Redis restart/eviction)
        3. current_equity (fail-open last resort)

        Both stores are updated when a new high is reached so they stay in sync.
        """
        peak = self._read_peak_equity()
        if current_equity > peak:
            self._write_peak_equity(current_equity)
            return current_equity
        return peak

    def _read_peak_equity(self) -> float:
        """Read peak equity from Redis, falling back to file backup."""
        # Try Redis first
        if self._redis:
            try:
                stored = self._redis.get(self.PEAK_EQUITY_KEY)
                if stored:
                    return float(stored)
            except (redis.RedisError, ValueError) as e:
                logger.warning(f"Redis error reading peak equity: {e}")

        # Fall back to file backup
        try:
            path = Path(self._PEAK_EQUITY_FILE)
            if path.exists():
                value = float(path.read_text().strip())
                logger.info(
                    f"Peak equity loaded from file backup: ${value:.2f}"
                )
                return value
        except (ValueError, OSError) as e:
            logger.warning(f"File backup error reading peak equity: {e}")

        return 0.0

    def _write_peak_equity(self, value: float) -> None:
        """Write peak equity to both Redis and file backup."""
        # Write to Redis
        if self._redis:
            try:
                self._redis.set(self.PEAK_EQUITY_KEY, str(value))
            except redis.RedisError as e:
                logger.warning(f"Redis error writing peak equity: {e}")

        # Write to file backup (atomic via temp file + rename)
        try:
            path = Path(self._PEAK_EQUITY_FILE)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(str(value))
            tmp.rename(path)
        except OSError as e:
            logger.warning(f"File backup error writing peak equity: {e}")
