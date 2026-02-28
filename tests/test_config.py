"""Tests for risk_engine.config — all config classes, load_settings, from_yaml."""

import os
from pathlib import Path

import pytest
import yaml

from risk_engine.config import (
    PortfolioRiskConfig,
    PositionSizingConfig,
    RiskSettings,
    TradeRiskConfig,
    VaRConfig,
    load_settings,
)


# ---------------------------------------------------------------------------
# PositionSizingConfig defaults
# ---------------------------------------------------------------------------


class TestPositionSizingConfig:
    def test_defaults(self):
        c = PositionSizingConfig()
        assert c.method == "atr"
        assert c.risk_per_trade_pct == 0.02
        assert c.atr_multiplier == 2.0
        assert c.max_position_pct == 0.20
        assert c.kelly_fraction == 0.25


# ---------------------------------------------------------------------------
# PortfolioRiskConfig defaults
# ---------------------------------------------------------------------------


class TestPortfolioRiskConfig:
    def test_defaults(self):
        c = PortfolioRiskConfig()
        assert c.max_concentration_pct == 0.20
        assert c.max_correlation == 0.80
        assert c.max_total_exposure_pct == 0.95
        assert c.min_cash_pct == 0.05
        assert c.max_open_positions == 5
        assert c.max_sector_exposure_pct == 0.40
        assert c.drawdown_threshold_pct == 0.10
        assert c.drawdown_size_reduction == 0.50


# ---------------------------------------------------------------------------
# VaRConfig defaults
# ---------------------------------------------------------------------------


class TestVaRConfig:
    def test_defaults(self):
        c = VaRConfig()
        assert c.enabled is True
        assert c.confidence == 0.95
        assert c.max_var_daily_pct == 0.05
        assert c.max_cvar_daily_pct == 0.08
        assert c.lookback_days == 252


# ---------------------------------------------------------------------------
# TradeRiskConfig defaults
# ---------------------------------------------------------------------------


class TestTradeRiskConfig:
    def test_defaults(self):
        c = TradeRiskConfig()
        assert c.max_atr_pct == 0.05
        assert c.require_above_sma200 is True
        assert c.min_volume_pct == 0.50


# ---------------------------------------------------------------------------
# RiskSettings defaults
# ---------------------------------------------------------------------------


class TestRiskSettingsDefaults:
    def test_redis_defaults(self):
        s = RiskSettings()
        assert s.redis_host == "localhost"
        assert s.redis_port == 6379
        assert s.redis_db == 0
        assert s.redis_positions_key == "robinhood:positions"

    def test_timescale_defaults(self):
        s = RiskSettings()
        assert s.timescale_host == "localhost"
        assert s.timescale_port == 5432
        assert s.timescale_db == "market_data"
        assert s.timescale_user == "postgres"
        assert s.timescale_password == ""

    def test_nested_configs_exist(self):
        s = RiskSettings()
        assert isinstance(s.position_sizing, PositionSizingConfig)
        assert isinstance(s.portfolio_risk, PortfolioRiskConfig)
        assert isinstance(s.var, VaRConfig)
        assert isinstance(s.trade_risk, TradeRiskConfig)

    def test_symbol_overrides_empty_default(self):
        s = RiskSettings()
        assert s.symbol_overrides == {}

    def test_log_level_default(self):
        s = RiskSettings()
        assert s.log_level == "INFO"


# ---------------------------------------------------------------------------
# RiskSettings — get_symbol_config
# ---------------------------------------------------------------------------


class TestGetSymbolConfig:
    def test_no_override_returns_default(self):
        s = RiskSettings()
        assert s.get_symbol_config("AAPL", "max_position_pct", 0.20) == 0.20

    def test_with_override(self):
        s = RiskSettings(
            symbol_overrides={"TSLA": {"max_position_pct": 0.10}}
        )
        assert s.get_symbol_config("TSLA", "max_position_pct", 0.20) == 0.10

    def test_override_for_wrong_symbol(self):
        s = RiskSettings(
            symbol_overrides={"TSLA": {"max_position_pct": 0.10}}
        )
        assert s.get_symbol_config("AAPL", "max_position_pct", 0.20) == 0.20

    def test_override_missing_key(self):
        s = RiskSettings(
            symbol_overrides={"TSLA": {"some_other_key": 42}}
        )
        assert s.get_symbol_config("TSLA", "max_position_pct", 0.20) == 0.20

    def test_none_default(self):
        s = RiskSettings()
        assert s.get_symbol_config("AAPL", "nonexistent") is None


# ---------------------------------------------------------------------------
# RiskSettings — sector groups
# ---------------------------------------------------------------------------


class TestSectorGroups:
    def test_default_empty(self):
        s = RiskSettings()
        assert s.sector_groups == {}

    def test_get_sector_for_symbol_found(self):
        s = RiskSettings(
            sector_groups={
                "uranium": ["CCJ", "URNM", "UUUU"],
                "precious_metals": ["WPM", "SLV"],
            }
        )
        assert s.get_sector_for_symbol("CCJ") == "uranium"
        assert s.get_sector_for_symbol("SLV") == "precious_metals"

    def test_get_sector_for_symbol_not_found(self):
        s = RiskSettings(
            sector_groups={"uranium": ["CCJ", "URNM"]}
        )
        assert s.get_sector_for_symbol("AAPL") is None

    def test_get_sector_empty_groups(self):
        s = RiskSettings()
        assert s.get_sector_for_symbol("CCJ") is None

    def test_sector_groups_from_yaml(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        config = {
            "sector_groups": {
                "uranium": ["CCJ", "URNM", "UUUU"],
                "precious_metals": ["WPM", "SLV"],
            }
        }
        yaml_path.write_text(yaml.dump(config))
        s = RiskSettings.from_yaml(str(yaml_path))
        assert s.get_sector_for_symbol("URNM") == "uranium"
        assert s.get_sector_for_symbol("WPM") == "precious_metals"
        assert len(s.sector_groups["uranium"]) == 3


# ---------------------------------------------------------------------------
# RiskSettings — get_max_position_pct
# ---------------------------------------------------------------------------


class TestGetMaxPositionPct:
    def test_default(self):
        s = RiskSettings()
        assert s.get_max_position_pct("AAPL") == 0.20

    def test_with_override(self):
        s = RiskSettings(
            symbol_overrides={"TSLA": {"max_position_pct": 0.15}}
        )
        assert s.get_max_position_pct("TSLA") == 0.15

    def test_no_override_falls_back(self):
        s = RiskSettings(
            symbol_overrides={"TSLA": {"max_position_pct": 0.15}}
        )
        assert s.get_max_position_pct("AAPL") == 0.20


# ---------------------------------------------------------------------------
# RiskSettings — env overrides
# ---------------------------------------------------------------------------


class TestRiskSettingsEnvOverrides:
    def test_override_redis_host(self, monkeypatch):
        monkeypatch.setenv("RISK_REDIS_HOST", "pi.local")
        s = RiskSettings()
        assert s.redis_host == "pi.local"

    def test_override_timescale_port(self, monkeypatch):
        monkeypatch.setenv("RISK_TIMESCALE_PORT", "5433")
        s = RiskSettings()
        assert s.timescale_port == 5433

    def test_override_log_level(self, monkeypatch):
        monkeypatch.setenv("RISK_LOG_LEVEL", "DEBUG")
        s = RiskSettings()
        assert s.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# RiskSettings — from_yaml
# ---------------------------------------------------------------------------


class TestFromYaml:
    def test_nonexistent_file(self, tmp_path):
        s = RiskSettings.from_yaml(str(tmp_path / "missing.yaml"))
        assert s.redis_host == "localhost"  # Defaults

    def test_empty_yaml(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("")
        s = RiskSettings.from_yaml(str(yaml_path))
        assert s.redis_host == "localhost"

    def test_partial_yaml(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump({"redis_host": "192.168.1.100"}))
        s = RiskSettings.from_yaml(str(yaml_path))
        assert s.redis_host == "192.168.1.100"
        assert s.redis_port == 6379  # Unchanged

    def test_nested_position_sizing(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        config = {
            "position_sizing": {
                "method": "kelly",
                "risk_per_trade_pct": 0.01,
            }
        }
        yaml_path.write_text(yaml.dump(config))
        s = RiskSettings.from_yaml(str(yaml_path))
        assert s.position_sizing.method == "kelly"
        assert s.position_sizing.risk_per_trade_pct == 0.01

    def test_nested_var_config(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        config = {"var": {"enabled": False, "confidence": 0.99}}
        yaml_path.write_text(yaml.dump(config))
        s = RiskSettings.from_yaml(str(yaml_path))
        assert s.var.enabled is False
        assert s.var.confidence == 0.99

    def test_symbol_overrides_in_yaml(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        config = {
            "symbol_overrides": {
                "TSLA": {"max_position_pct": 0.10},
                "GME": {"max_position_pct": 0.05},
            }
        }
        yaml_path.write_text(yaml.dump(config))
        s = RiskSettings.from_yaml(str(yaml_path))
        assert s.symbol_overrides["TSLA"]["max_position_pct"] == 0.10
        assert s.symbol_overrides["GME"]["max_position_pct"] == 0.05

    def test_full_yaml(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        config = {
            "redis_host": "redis.local",
            "redis_port": 6380,
            "timescale_host": "ts.local",
            "log_level": "DEBUG",
            "position_sizing": {"method": "fixed", "risk_per_trade_pct": 0.03},
            "portfolio_risk": {"max_concentration_pct": 0.15},
            "var": {"enabled": True, "lookback_days": 126},
            "trade_risk": {"max_atr_pct": 0.08},
        }
        yaml_path.write_text(yaml.dump(config))
        s = RiskSettings.from_yaml(str(yaml_path))
        assert s.redis_host == "redis.local"
        assert s.redis_port == 6380
        assert s.position_sizing.method == "fixed"
        assert s.portfolio_risk.max_concentration_pct == 0.15
        assert s.var.lookback_days == 126
        assert s.trade_risk.max_atr_pct == 0.08


# ---------------------------------------------------------------------------
# load_settings
# ---------------------------------------------------------------------------


class TestLoadSettings:
    def test_no_config_returns_defaults(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        s = load_settings()
        assert s.redis_host == "localhost"

    def test_explicit_path(self, tmp_path):
        yaml_path = tmp_path / "risk.yaml"
        yaml_path.write_text(yaml.dump({"redis_host": "custom.host"}))
        s = load_settings(str(yaml_path))
        assert s.redis_host == "custom.host"

    def test_searches_standard_locations(self, monkeypatch, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        yaml_path = config_dir / "risk_config.yaml"
        yaml_path.write_text(yaml.dump({"log_level": "WARN"}))
        monkeypatch.chdir(tmp_path)
        s = load_settings()
        assert s.log_level == "WARN"

    def test_none_path_no_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        s = load_settings(None)
        assert isinstance(s, RiskSettings)
