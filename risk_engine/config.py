"""
Risk engine configuration.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class PositionSizingConfig(BaseSettings):
    """Position sizing configuration."""

    method: str = "atr"
    risk_per_trade_pct: float = 0.02
    atr_multiplier: float = 2.0
    max_position_pct: float = 0.20
    kelly_fraction: float = 0.25


class PortfolioRiskConfig(BaseSettings):
    """Portfolio risk limits configuration."""

    max_concentration_pct: float = 0.20
    max_correlation: float = 0.80
    max_total_exposure_pct: float = 0.95
    min_cash_pct: float = 0.05
    max_open_positions: int = 5
    max_sector_exposure_pct: float = 0.40
    max_portfolio_heat_pct: float = 0.08
    max_sector_positions: int = 2
    drawdown_threshold_pct: float = 0.10
    drawdown_size_reduction: float = 0.50
    drawdown_halt_pct: float = 0.15


class VaRConfig(BaseSettings):
    """Value at Risk configuration."""

    enabled: bool = True
    confidence: float = 0.95
    max_var_daily_pct: float = 0.05
    max_cvar_daily_pct: float = 0.08
    lookback_days: int = 252


class CapitalTemperatureConfig(BaseSettings):
    """Capital temperature gauge configuration."""

    enabled: bool = True
    max_reduction: float = 0.30
    vix_weight: float = 0.60
    hy_spread_weight: float = 0.40
    vix_floor: float = 15.0
    vix_ceiling: float = 40.0
    hy_floor: float = 3.5
    hy_ceiling: float = 8.0


class TradeRiskConfig(BaseSettings):
    """Trade-level risk configuration."""

    max_atr_pct: float = 0.05
    require_above_sma200: bool = True
    min_volume_pct: float = 0.50


class RiskSettings(BaseSettings):
    """Main risk engine settings."""

    # Redis configuration
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_positions_key: str = Field(default="robinhood:positions")

    # TimescaleDB configuration
    timescale_host: str = Field(default="localhost")
    timescale_port: int = Field(default=5432)
    timescale_db: str = Field(default="market_data")
    timescale_user: str = Field(default="postgres")
    timescale_password: str = Field(default="")

    # Risk module configs (nested)
    position_sizing: PositionSizingConfig = Field(
        default_factory=PositionSizingConfig
    )
    portfolio_risk: PortfolioRiskConfig = Field(
        default_factory=PortfolioRiskConfig
    )
    var: VaRConfig = Field(default_factory=VaRConfig)
    trade_risk: TradeRiskConfig = Field(default_factory=TradeRiskConfig)
    capital_temperature: CapitalTemperatureConfig = Field(
        default_factory=CapitalTemperatureConfig
    )

    # Symbol-specific overrides
    symbol_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Sector groupings for concentration limits
    sector_groups: Dict[str, List[str]] = Field(default_factory=dict)

    # Logging
    log_level: str = Field(default="INFO")

    model_config = ConfigDict(
        env_prefix="RISK_",
        env_nested_delimiter="__",
    )

    def get_symbol_config(
        self, symbol: str, key: str, default: Any = None
    ) -> Any:
        """Get a config value with symbol-specific override."""
        if symbol in self.symbol_overrides:
            override = self.symbol_overrides[symbol]
            if key in override:
                return override[key]
        return default

    def get_sector_for_symbol(self, symbol: str) -> Optional[str]:
        """Look up which sector a symbol belongs to."""
        for sector, symbols in self.sector_groups.items():
            if symbol in symbols:
                return sector
        return None

    def get_max_position_pct(self, symbol: str) -> float:
        """Get max position percentage for a symbol."""
        return self.get_symbol_config(
            symbol,
            "max_position_pct",
            self.position_sizing.max_position_pct,
        )

    def get_risk_per_trade_pct(self, symbol: str) -> float:
        """Get risk per trade for a symbol (check overrides, then default)."""
        return self.get_symbol_config(
            symbol,
            "risk_per_trade_pct",
            self.position_sizing.risk_per_trade_pct,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RiskSettings":
        """Load settings from YAML file with environment overrides."""
        config_dict = {}

        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}

            # Flatten nested configs for pydantic
            config_dict = cls._flatten_yaml_config(yaml_config)

        return cls(**config_dict)

    @classmethod
    def _flatten_yaml_config(cls, yaml_config: Dict) -> Dict:
        """Convert nested YAML to flat dict with nested models."""
        result = {}

        # Direct mappings
        for key in [
            "redis_host",
            "redis_port",
            "redis_db",
            "redis_positions_key",
            "timescale_host",
            "timescale_port",
            "timescale_db",
            "timescale_user",
            "timescale_password",
            "log_level",
        ]:
            if key in yaml_config:
                result[key] = yaml_config[key]

        # Nested configs
        if "position_sizing" in yaml_config:
            result["position_sizing"] = PositionSizingConfig(
                **yaml_config["position_sizing"]
            )

        if "portfolio_risk" in yaml_config:
            result["portfolio_risk"] = PortfolioRiskConfig(
                **yaml_config["portfolio_risk"]
            )

        if "var" in yaml_config:
            result["var"] = VaRConfig(**yaml_config["var"])

        if "trade_risk" in yaml_config:
            result["trade_risk"] = TradeRiskConfig(**yaml_config["trade_risk"])

        if "capital_temperature" in yaml_config:
            result["capital_temperature"] = CapitalTemperatureConfig(
                **yaml_config["capital_temperature"]
            )

        if "symbol_overrides" in yaml_config:
            result["symbol_overrides"] = yaml_config["symbol_overrides"]

        if "sector_groups" in yaml_config:
            result["sector_groups"] = yaml_config["sector_groups"]

        return result


def load_settings(config_path: Optional[str] = None) -> RiskSettings:
    """
    Load risk settings.

    Priority:
    1. Environment variables (highest)
    2. YAML config file
    3. Defaults (lowest)
    """
    if config_path is None:
        # Look for config in standard locations
        search_paths = [
            Path("config/risk_config.yaml"),
            Path("/etc/risk-engine/risk_config.yaml"),
            Path.home() / ".config" / "risk-engine" / "risk_config.yaml",
        ]
        for path in search_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path:
        return RiskSettings.from_yaml(config_path)

    return RiskSettings()
