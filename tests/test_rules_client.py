"""Tests for risk_engine.data.rules_client — RulesClient with mocked Redis."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import redis

from risk_engine.config import RiskSettings
from risk_engine.data.rules_client import (
    EXIT_STRATEGY_KEY,
    RULES_CONFIG_KEY,
    RULES_UPDATED_KEY,
    SYMBOL_RULES_PREFIX,
    RulesClient,
)


@pytest.fixture
def mock_settings():
    return RiskSettings()


@pytest.fixture
def client(mock_settings):
    return RulesClient(mock_settings)


# ---------------------------------------------------------------------------
# connect / close
# ---------------------------------------------------------------------------


class TestConnect:
    def test_connect_success(self, client):
        with patch.object(redis, "Redis") as MockRedis:
            instance = MockRedis.return_value
            instance.ping.return_value = True
            result = client.connect()
            assert result is True
            assert client._redis is not None

    def test_connect_failure(self, client):
        with patch.object(redis, "Redis") as MockRedis:
            MockRedis.return_value.ping.side_effect = redis.RedisError("refused")
            result = client.connect()
            assert result is False

    def test_close_clears_connection(self, client):
        client._redis = MagicMock()
        client.close()
        assert client._redis is None

    def test_close_when_not_connected(self, client):
        client.close()  # Should not raise


# ---------------------------------------------------------------------------
# get_exit_strategy
# ---------------------------------------------------------------------------


class TestGetExitStrategy:
    def test_no_redis_returns_default(self, client):
        result = client.get_exit_strategy("AAPL")
        assert result == {"profit_target": 0.07, "stop_loss": 0.05}

    def test_symbol_specific_override(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        symbol_data = json.dumps(
            {"exit_strategy": {"profit_target": 0.10, "stop_loss": 0.08}}
        )
        mock_redis.get.side_effect = lambda key: (
            symbol_data if key == f"{SYMBOL_RULES_PREFIX}TSLA" else None
        )
        result = client.get_exit_strategy("TSLA")
        assert result == {"profit_target": 0.10, "stop_loss": 0.08}

    def test_falls_back_to_default_strategy(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        default_data = json.dumps({"profit_target": 0.12, "stop_loss": 0.06})
        mock_redis.get.side_effect = lambda key: (
            default_data if key == EXIT_STRATEGY_KEY else None
        )
        result = client.get_exit_strategy("AAPL")
        assert result == {"profit_target": 0.12, "stop_loss": 0.06}

    def test_no_symbol_no_default_returns_hardcoded(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        mock_redis.get.return_value = None
        result = client.get_exit_strategy("AAPL")
        assert result == {"profit_target": 0.07, "stop_loss": 0.05}

    def test_redis_error_returns_default(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        mock_redis.get.side_effect = redis.RedisError("timeout")
        result = client.get_exit_strategy("AAPL")
        assert result == {"profit_target": 0.07, "stop_loss": 0.05}

    def test_json_decode_error_returns_default(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        mock_redis.get.return_value = "not-valid-json"
        result = client.get_exit_strategy("AAPL")
        assert result == {"profit_target": 0.07, "stop_loss": 0.05}

    def test_symbol_data_no_exit_strategy_falls_through(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        # Symbol data exists but has no exit_strategy key
        mock_redis.get.side_effect = lambda key: (
            json.dumps({"some_other": "data"})
            if key == f"{SYMBOL_RULES_PREFIX}AAPL"
            else None
        )
        result = client.get_exit_strategy("AAPL")
        assert result == {"profit_target": 0.07, "stop_loss": 0.05}


# ---------------------------------------------------------------------------
# get_stop_loss_pct / get_profit_target_pct
# ---------------------------------------------------------------------------


class TestHelperMethods:
    def test_get_stop_loss_pct_default(self, client):
        assert client.get_stop_loss_pct("AAPL") == 0.05

    def test_get_profit_target_pct_default(self, client):
        assert client.get_profit_target_pct("AAPL") == 0.07

    def test_get_stop_loss_from_redis(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        mock_redis.get.side_effect = lambda key: (
            json.dumps({"profit_target": 0.10, "stop_loss": 0.03})
            if key == EXIT_STRATEGY_KEY
            else None
        )
        assert client.get_stop_loss_pct("AAPL") == 0.03

    def test_get_profit_target_from_redis(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        mock_redis.get.side_effect = lambda key: (
            json.dumps({"profit_target": 0.15, "stop_loss": 0.05})
            if key == EXIT_STRATEGY_KEY
            else None
        )
        assert client.get_profit_target_pct("AAPL") == 0.15


# ---------------------------------------------------------------------------
# get_config — caching
# ---------------------------------------------------------------------------


class TestGetConfig:
    def test_no_redis_returns_cache(self, client):
        client._config_cache = {"rules": []}
        assert client.get_config() == {"rules": []}

    def test_no_redis_no_cache(self, client):
        assert client.get_config() is None

    def test_loads_from_redis(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        config = {"rules": ["buy_dip"], "version": 2}
        mock_redis.get.return_value = json.dumps(config)
        result = client.get_config()
        assert result == config
        assert client._config_cache == config
        assert client._cache_time is not None

    def test_uses_cache_within_ttl(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        client._config_cache = {"cached": True}
        client._cache_time = datetime.utcnow()
        result = client.get_config()
        assert result == {"cached": True}
        mock_redis.get.assert_not_called()

    def test_force_refresh_ignores_cache(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        client._config_cache = {"old": True}
        client._cache_time = datetime.utcnow()
        mock_redis.get.return_value = json.dumps({"new": True})
        result = client.get_config(force_refresh=True)
        assert result == {"new": True}

    def test_redis_error_returns_cached(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        client._config_cache = {"fallback": True}
        client._cache_time = None  # Force refresh
        mock_redis.get.side_effect = redis.RedisError("fail")
        result = client.get_config()
        assert result == {"fallback": True}

    def test_no_data_in_redis(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        mock_redis.get.return_value = None
        assert client.get_config() is None


# ---------------------------------------------------------------------------
# get_last_updated
# ---------------------------------------------------------------------------


class TestGetLastUpdated:
    def test_no_redis(self, client):
        assert client.get_last_updated() is None

    def test_valid_timestamp(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        mock_redis.get.return_value = "2026-02-20T10:30:00"
        result = client.get_last_updated()
        assert result == datetime(2026, 2, 20, 10, 30, 0)

    def test_invalid_timestamp(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        mock_redis.get.return_value = "not-a-date"
        assert client.get_last_updated() is None

    def test_redis_error(self, client):
        mock_redis = MagicMock()
        client._redis = mock_redis
        mock_redis.get.side_effect = redis.RedisError("timeout")
        assert client.get_last_updated() is None
