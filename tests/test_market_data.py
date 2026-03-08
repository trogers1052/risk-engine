"""Tests for risk_engine.data.market_data — MarketDataLoader with mocked psycopg2."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from risk_engine.config import RiskSettings
from risk_engine.data.market_data import MarketDataLoader


@pytest.fixture
def mock_settings():
    return RiskSettings()


@pytest.fixture
def loader(mock_settings):
    return MarketDataLoader(mock_settings)


# ---------------------------------------------------------------------------
# connect / close
# ---------------------------------------------------------------------------


class TestConnect:
    @patch("risk_engine.data.market_data.psycopg2")
    def test_connect_success(self, mock_psycopg2, loader):
        result = loader.connect()
        assert result is True
        assert loader._conn is not None

    @patch("risk_engine.data.market_data.psycopg2")
    def test_connect_failure(self, mock_psycopg2, loader):
        import psycopg2 as real_psycopg2
        mock_psycopg2.connect.side_effect = real_psycopg2.Error("refused")
        mock_psycopg2.Error = real_psycopg2.Error
        result = loader.connect()
        assert result is False

    def test_close(self, loader):
        loader._conn = MagicMock()
        loader.close()
        assert loader._conn is None

    def test_close_when_not_connected(self, loader):
        loader.close()  # Should not raise


# ---------------------------------------------------------------------------
# get_daily_prices
# ---------------------------------------------------------------------------


class TestGetDailyPrices:
    def test_not_connected_returns_none(self, loader):
        assert loader.get_daily_prices("AAPL") is None

    def test_returns_series(self, loader):
        mock_conn = MagicMock()
        loader._conn = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        rows = [
            {"date": datetime(2023, 1, 2), "close": 150.0},
            {"date": datetime(2023, 1, 3), "close": 152.0},
            {"date": datetime(2023, 1, 4), "close": 148.0},
        ]
        mock_cursor.fetchall.return_value = rows
        result = loader.get_daily_prices("AAPL", lookback_days=10)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_no_rows_returns_none(self, loader):
        mock_conn = MagicMock()
        loader._conn = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        assert loader.get_daily_prices("AAPL") is None

    def test_db_error_returns_none(self, loader):
        mock_conn = MagicMock()
        loader._conn = mock_conn
        import psycopg2
        mock_conn.cursor.side_effect = psycopg2.Error("query failed")
        assert loader.get_daily_prices("AAPL") is None


# ---------------------------------------------------------------------------
# get_daily_returns
# ---------------------------------------------------------------------------


class TestGetDailyReturns:
    def test_calculates_pct_change(self, loader):
        """Returns are percent changes of prices."""
        loader.get_daily_prices = MagicMock(return_value=pd.Series(
            [100.0, 110.0, 105.0],
            index=pd.date_range("2023-01-01", periods=3, freq="D"),
        ))
        result = loader.get_daily_returns("AAPL")
        assert len(result) == 2
        assert result.iloc[0] == pytest.approx(0.10)
        assert result.iloc[1] == pytest.approx(-0.04545, rel=1e-3)

    def test_no_prices_returns_none(self, loader):
        loader.get_daily_prices = MagicMock(return_value=None)
        assert loader.get_daily_returns("AAPL") is None

    def test_single_price_returns_none(self, loader):
        loader.get_daily_prices = MagicMock(return_value=pd.Series([100.0]))
        assert loader.get_daily_returns("AAPL") is None


# ---------------------------------------------------------------------------
# get_multiple_returns
# ---------------------------------------------------------------------------


class TestGetMultipleReturns:
    def test_empty_symbols(self, loader):
        assert loader.get_multiple_returns([]) is None

    def test_all_symbols_fail(self, loader):
        loader.get_daily_returns = MagicMock(return_value=None)
        assert loader.get_multiple_returns(["AAPL", "GOOG"]) is None

    def test_returns_dataframe(self, loader):
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        loader.get_daily_returns = MagicMock(side_effect=[
            pd.Series([0.01, 0.02, -0.01, 0.005, 0.03], index=dates),
            pd.Series([0.02, -0.01, 0.015, 0.01, -0.005], index=dates),
        ])
        result = loader.get_multiple_returns(["AAPL", "GOOG"])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["AAPL", "GOOG"]
        assert len(result) == 5


# ---------------------------------------------------------------------------
# get_correlation_matrix
# ---------------------------------------------------------------------------


class TestGetCorrelationMatrix:
    def test_returns_correlation(self, loader):
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        returns = pd.DataFrame({
            "AAPL": np.random.randn(20) * 0.02,
            "GOOG": np.random.randn(20) * 0.02,
        }, index=dates)
        loader.get_multiple_returns = MagicMock(return_value=returns)
        result = loader.get_correlation_matrix(["AAPL", "GOOG"])
        assert result is not None
        assert result.shape == (2, 2)
        assert result.loc["AAPL", "AAPL"] == pytest.approx(1.0)

    def test_insufficient_data(self, loader):
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        returns = pd.DataFrame({"AAPL": [0.01] * 5}, index=dates)
        loader.get_multiple_returns = MagicMock(return_value=returns)
        result = loader.get_correlation_matrix(["AAPL"])
        assert result is None  # < 10 rows

    def test_no_data(self, loader):
        loader.get_multiple_returns = MagicMock(return_value=None)
        assert loader.get_correlation_matrix(["AAPL"]) is None


# ---------------------------------------------------------------------------
# get_correlation_with_portfolio
# ---------------------------------------------------------------------------


class TestGetCorrelationWithPortfolio:
    def test_returns_correlations(self, loader):
        corr = pd.DataFrame(
            [[1.0, 0.75, 0.3], [0.75, 1.0, 0.5], [0.3, 0.5, 1.0]],
            index=["NEW", "AAPL", "GOOG"],
            columns=["NEW", "AAPL", "GOOG"],
        )
        loader.get_correlation_matrix = MagicMock(return_value=corr)
        result = loader.get_correlation_with_portfolio("NEW", ["AAPL", "GOOG"])
        assert result["AAPL"] == 0.75
        assert result["GOOG"] == 0.3

    def test_no_correlation_data(self, loader):
        loader.get_correlation_matrix = MagicMock(return_value=None)
        assert loader.get_correlation_with_portfolio("NEW", ["AAPL"]) == {}

    def test_symbol_not_in_matrix(self, loader):
        corr = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]],
            index=["AAPL", "GOOG"],
            columns=["AAPL", "GOOG"],
        )
        loader.get_correlation_matrix = MagicMock(return_value=corr)
        assert loader.get_correlation_with_portfolio("MSFT", ["AAPL"]) == {}


# ---------------------------------------------------------------------------
# get_max_correlation_with_portfolio
# ---------------------------------------------------------------------------


class TestGetMaxCorrelation:
    def test_finds_max(self, loader):
        loader.get_correlation_with_portfolio = MagicMock(
            return_value={"AAPL": 0.80, "GOOG": 0.30, "MSFT": -0.90}
        )
        symbol, corr = loader.get_max_correlation_with_portfolio("NEW", ["AAPL", "GOOG", "MSFT"])
        # max abs correlation is MSFT at -0.90
        assert symbol == "MSFT"
        assert corr == -0.90

    def test_empty_correlations(self, loader):
        loader.get_correlation_with_portfolio = MagicMock(return_value={})
        symbol, corr = loader.get_max_correlation_with_portfolio("NEW", ["AAPL"])
        assert symbol is None
        assert corr == 0.0


# ---------------------------------------------------------------------------
# get_average_volume
# ---------------------------------------------------------------------------


class TestGetAverageVolume:
    def test_not_connected(self, loader):
        assert loader.get_average_volume("AAPL") is None

    def test_returns_average(self, loader):
        mock_conn = MagicMock()
        loader._conn = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            {"daily_volume": 1_000_000},
            {"daily_volume": 1_200_000},
            {"daily_volume": 800_000},
        ]
        result = loader.get_average_volume("AAPL")
        assert result == pytest.approx(1_000_000.0)

    def test_no_rows(self, loader):
        mock_conn = MagicMock()
        loader._conn = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        assert loader.get_average_volume("AAPL") is None

    def test_all_null_volumes(self, loader):
        mock_conn = MagicMock()
        loader._conn = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            {"daily_volume": None},
            {"daily_volume": None},
        ]
        assert loader.get_average_volume("AAPL") is None


# ---------------------------------------------------------------------------
# get_portfolio_returns
# ---------------------------------------------------------------------------


class TestGetPortfolioReturns:
    def test_calculates_weighted_returns(self, loader):
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        returns_df = pd.DataFrame({
            "AAPL": [0.01, 0.02, -0.01, 0.005, 0.03],
            "GOOG": [0.02, -0.01, 0.015, 0.01, -0.005],
        }, index=dates)
        loader.get_multiple_returns = MagicMock(return_value=returns_df)
        result = loader.get_portfolio_returns({"AAPL": 0.6, "GOOG": 0.4})
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_no_returns_data(self, loader):
        loader.get_multiple_returns = MagicMock(return_value=None)
        assert loader.get_portfolio_returns({"AAPL": 1.0}) is None
