"""Tests for risk_engine.data.portfolio_state — PortfolioStateManager with mocked Redis."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import redis

from risk_engine.config import RiskSettings
from risk_engine.data.portfolio_state import PortfolioStateManager
from risk_engine.models.portfolio import PortfolioState


@pytest.fixture
def mock_settings():
    return RiskSettings()


@pytest.fixture
def manager(mock_settings):
    return PortfolioStateManager(mock_settings)


# ---------------------------------------------------------------------------
# connect / close
# ---------------------------------------------------------------------------


class TestConnect:
    def test_connect_success(self, manager):
        with patch.object(redis, "Redis") as MockRedis:
            MockRedis.return_value.ping.return_value = True
            assert manager.connect() is True
            assert manager._redis is not None

    def test_connect_failure(self, manager):
        with patch.object(redis, "Redis") as MockRedis:
            MockRedis.return_value.ping.side_effect = redis.RedisError("refused")
            assert manager.connect() is False

    def test_close(self, manager):
        manager._redis = MagicMock()
        manager.close()
        assert manager._redis is None

    def test_close_when_not_connected(self, manager):
        manager.close()  # Should not raise


# ---------------------------------------------------------------------------
# _load_from_redis
# ---------------------------------------------------------------------------


class TestLoadFromRedis:
    def test_no_redis_returns_empty(self, manager):
        state = manager._load_from_redis()
        assert isinstance(state, PortfolioState)
        assert len(state.positions) == 0

    def test_no_data_in_redis(self, manager):
        manager._redis = MagicMock()
        manager._redis.get.return_value = None
        state = manager._load_from_redis()
        assert len(state.positions) == 0

    def test_valid_portfolio_data(self, manager):
        manager._redis = MagicMock()
        data = {
            "positions": [
                {
                    "symbol": "AAPL",
                    "quantity": 100,
                    "average_buy_price": 150.0,
                    "current_price": 160.0,
                    "market_value": 16000.0,
                    "unrealized_pnl": 1000.0,
                    "unrealized_pnl_pct": 0.0667,
                },
            ],
            "account": {
                "buying_power": 10000.0,
                "cash": 10000.0,
                "total_equity": 26000.0,
                "market_value": 16000.0,
            },
            "timestamp": "2026-02-20T10:00:00",
        }
        manager._redis.get.return_value = json.dumps(data)
        state = manager._load_from_redis()
        assert "AAPL" in state.positions
        assert state.positions["AAPL"].shares == 100
        assert state.positions["AAPL"].current_price == 160.0
        assert state.buying_power == 10000.0
        assert state.total_equity == 26000.0
        assert state.last_updated is not None

    def test_position_missing_symbol_skipped(self, manager):
        manager._redis = MagicMock()
        data = {
            "positions": [
                {"quantity": 50, "current_price": 100.0},  # No symbol
                {"symbol": "GOOG", "quantity": 25, "current_price": 200.0},
            ],
            "account": {},
        }
        manager._redis.get.return_value = json.dumps(data)
        state = manager._load_from_redis()
        assert len(state.positions) == 1
        assert "GOOG" in state.positions

    def test_position_invalid_values(self, manager):
        manager._redis = MagicMock()
        data = {
            "positions": [
                {"symbol": "BAD", "quantity": "not_a_number"},
            ],
            "account": {},
        }
        manager._redis.get.return_value = json.dumps(data)
        state = manager._load_from_redis()
        # Invalid parse → skipped
        assert len(state.positions) == 0

    def test_redis_error_returns_cache(self, manager):
        manager._redis = MagicMock()
        manager._redis.get.side_effect = redis.RedisError("fail")
        cached = PortfolioState(buying_power=5000)
        manager._cache = cached
        state = manager._load_from_redis()
        assert state.buying_power == 5000

    def test_redis_error_no_cache(self, manager):
        manager._redis = MagicMock()
        manager._redis.get.side_effect = redis.RedisError("fail")
        state = manager._load_from_redis()
        assert isinstance(state, PortfolioState)
        assert len(state.positions) == 0

    def test_invalid_json(self, manager):
        manager._redis = MagicMock()
        manager._redis.get.return_value = "not-json"
        state = manager._load_from_redis()
        assert isinstance(state, PortfolioState)

    def test_timestamp_with_z_suffix(self, manager):
        manager._redis = MagicMock()
        data = {
            "positions": [],
            "account": {},
            "timestamp": "2026-02-20T10:00:00Z",
        }
        manager._redis.get.return_value = json.dumps(data)
        state = manager._load_from_redis()
        assert state.last_updated is not None

    def test_invalid_timestamp(self, manager):
        manager._redis = MagicMock()
        data = {
            "positions": [],
            "account": {},
            "timestamp": "not-a-date",
        }
        manager._redis.get.return_value = json.dumps(data)
        state = manager._load_from_redis()
        assert state.last_updated is None

    def test_last_updated_key(self, manager):
        manager._redis = MagicMock()
        data = {
            "positions": [],
            "account": {},
            "last_updated": "2026-01-15T08:30:00",
        }
        manager._redis.get.return_value = json.dumps(data)
        state = manager._load_from_redis()
        assert state.last_updated is not None


# ---------------------------------------------------------------------------
# get_portfolio_state — caching
# ---------------------------------------------------------------------------


class TestGetPortfolioState:
    def test_returns_cached_within_ttl(self, manager):
        cached = PortfolioState(buying_power=9999)
        manager._cache = cached
        manager._cache_time = datetime.utcnow()
        result = manager.get_portfolio_state()
        assert result.buying_power == 9999

    def test_force_refresh_ignores_cache(self, manager):
        manager._redis = MagicMock()
        manager._redis.get.return_value = json.dumps({
            "positions": [],
            "account": {"buying_power": 1234},
        })
        manager._cache = PortfolioState(buying_power=9999)
        manager._cache_time = datetime.utcnow()
        result = manager.get_portfolio_state(force_refresh=True)
        assert result.buying_power == 1234

    def test_refreshes_when_no_cache(self, manager):
        manager._redis = MagicMock()
        manager._redis.get.return_value = json.dumps({
            "positions": [],
            "account": {"buying_power": 5000},
        })
        result = manager.get_portfolio_state()
        assert result.buying_power == 5000
        assert manager._cache is not None
        assert manager._cache_time is not None


# ---------------------------------------------------------------------------
# convenience methods
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    def test_get_position_weight(self, manager, sample_portfolio):
        manager._cache = sample_portfolio
        manager._cache_time = datetime.utcnow()
        weight = manager.get_position_weight("AAPL")
        assert weight > 0

    def test_get_buying_power(self, manager, sample_portfolio):
        manager._cache = sample_portfolio
        manager._cache_time = datetime.utcnow()
        assert manager.get_buying_power() == 10000.0

    def test_get_total_equity(self, manager, sample_portfolio):
        manager._cache = sample_portfolio
        manager._cache_time = datetime.utcnow()
        assert manager.get_total_equity() == 31500.0

    def test_has_position(self, manager, sample_portfolio):
        manager._cache = sample_portfolio
        manager._cache_time = datetime.utcnow()
        assert manager.has_position("AAPL") is True
        assert manager.has_position("MSFT") is False


# ---------------------------------------------------------------------------
# Peak equity tracking with file backup
# ---------------------------------------------------------------------------


class TestPeakEquity:
    def test_new_high_updates_redis_and_file(self, manager, tmp_path):
        """New equity high should write to both Redis and file backup."""
        peak_file = tmp_path / "peak_equity"
        manager._PEAK_EQUITY_FILE = str(peak_file)
        manager._redis = MagicMock()
        manager._redis.get.return_value = "900.0"

        result = manager.get_peak_equity(1000.0)
        assert result == 1000.0
        manager._redis.set.assert_called_once_with("risk:peak_equity", "1000.0")
        assert peak_file.exists()
        assert float(peak_file.read_text().strip()) == 1000.0

    def test_below_peak_returns_stored(self, manager, tmp_path):
        """Equity below peak should return the stored peak, not current."""
        peak_file = tmp_path / "peak_equity"
        manager._PEAK_EQUITY_FILE = str(peak_file)
        manager._redis = MagicMock()
        manager._redis.get.return_value = "1200.0"

        result = manager.get_peak_equity(1000.0)
        assert result == 1200.0
        manager._redis.set.assert_not_called()

    def test_redis_down_falls_back_to_file(self, manager, tmp_path):
        """When Redis is unavailable, should read peak from file backup."""
        peak_file = tmp_path / "peak_equity"
        peak_file.write_text("1500.0")
        manager._PEAK_EQUITY_FILE = str(peak_file)
        manager._redis = MagicMock()
        manager._redis.get.side_effect = redis.RedisError("connection refused")

        result = manager.get_peak_equity(1000.0)
        assert result == 1500.0

    def test_redis_down_new_high_still_writes_file(self, manager, tmp_path):
        """When Redis is down but equity is new high, file should update."""
        peak_file = tmp_path / "peak_equity"
        peak_file.write_text("800.0")
        manager._PEAK_EQUITY_FILE = str(peak_file)
        manager._redis = MagicMock()
        manager._redis.get.side_effect = redis.RedisError("connection refused")
        manager._redis.set.side_effect = redis.RedisError("connection refused")

        result = manager.get_peak_equity(1000.0)
        assert result == 1000.0
        assert float(peak_file.read_text().strip()) == 1000.0

    def test_no_redis_no_file_returns_zero(self, manager, tmp_path):
        """With no Redis and no file, _read_peak_equity returns 0.0."""
        manager._PEAK_EQUITY_FILE = str(tmp_path / "nonexistent")
        manager._redis = None

        result = manager.get_peak_equity(500.0)
        # 500 > 0.0, so it's a new high
        assert result == 500.0

    def test_corrupt_file_ignored(self, manager, tmp_path):
        """Corrupt file backup should be ignored gracefully."""
        peak_file = tmp_path / "peak_equity"
        peak_file.write_text("not-a-number")
        manager._PEAK_EQUITY_FILE = str(peak_file)
        manager._redis = MagicMock()
        manager._redis.get.return_value = None

        result = manager.get_peak_equity(600.0)
        # File corrupt, Redis empty → peak is 0.0, so 600 is new high
        assert result == 600.0

    def test_redis_empty_reads_file(self, manager, tmp_path):
        """When Redis key doesn't exist, should fall back to file."""
        peak_file = tmp_path / "peak_equity"
        peak_file.write_text("2000.0")
        manager._PEAK_EQUITY_FILE = str(peak_file)
        manager._redis = MagicMock()
        manager._redis.get.return_value = None  # Key doesn't exist

        result = manager.get_peak_equity(1800.0)
        assert result == 2000.0

    def test_file_directory_created_automatically(self, manager, tmp_path):
        """File backup should create parent directories if needed."""
        peak_file = tmp_path / "subdir" / "deep" / "peak_equity"
        manager._PEAK_EQUITY_FILE = str(peak_file)
        manager._redis = MagicMock()
        manager._redis.get.return_value = None

        manager.get_peak_equity(750.0)
        assert peak_file.exists()
        assert float(peak_file.read_text().strip()) == 750.0
