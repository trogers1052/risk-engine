"""Tests for alert publisher."""

import pytest
from unittest.mock import MagicMock, patch

from risk_engine.alerts.publisher import AlertPublisher, AlertType


class TestAlertPublisher:
    """Tests for AlertPublisher."""

    def test_init(self):
        """Test initialization."""
        publisher = AlertPublisher(
            brokers=["localhost:9092"],
            topic="trading.risk-alerts",
        )

        assert publisher.brokers == ["localhost:9092"]
        assert publisher.topic == "trading.risk-alerts"

    @patch("risk_engine.alerts.publisher.HAS_KAFKA", False)
    def test_connect_no_kafka(self):
        """Test connect when kafka-python not installed."""
        # Since we're testing without kafka, connect should still succeed
        publisher = AlertPublisher(brokers=["localhost:9092"])

        # Should return True (graceful degradation)
        result = publisher.connect()
        assert result is True

    @patch("risk_engine.alerts.publisher.HAS_KAFKA", True)
    @patch("risk_engine.alerts.publisher.KafkaProducer")
    def test_connect_with_kafka(self, mock_producer_class):
        """Test connect when kafka-python is installed."""
        mock_producer = MagicMock()
        mock_producer_class.return_value = mock_producer

        publisher = AlertPublisher(brokers=["localhost:9092"])
        result = publisher.connect()

        assert result is True
        mock_producer_class.assert_called_once()

    def test_publish_stop_loss_alert_no_kafka(self):
        """Test publishing stop loss alert without Kafka."""
        publisher = AlertPublisher(brokers=["localhost:9092"])
        publisher.connect()

        result = publisher.publish_stop_loss_alert(
            symbol="AAPL",
            shares=100,
            average_cost=150.00,
            stop_loss_price=142.50,
            stop_loss_pct=0.05,
            current_price=155.00,
        )

        # Should log the alert even without Kafka
        assert result is True

    def test_publish_stop_triggered_alert_no_kafka(self):
        """Test publishing stop triggered alert without Kafka."""
        publisher = AlertPublisher(brokers=["localhost:9092"])
        publisher.connect()

        result = publisher.publish_stop_triggered_alert(
            symbol="GOOGL",
            shares=50,
            average_cost=140.00,
            stop_loss_price=133.00,
            current_price=130.00,
            loss_amount=350.00,
        )

        assert result is True

    def test_publish_stop_approaching_alert(self):
        """Test publishing stop approaching alert."""
        publisher = AlertPublisher(brokers=["localhost:9092"])
        publisher.connect()

        result = publisher.publish_stop_approaching_alert(
            symbol="MSFT",
            current_price=395.00,
            stop_loss_price=380.00,
            distance_pct=0.04,
        )

        assert result is True

    def test_publish_generic_risk_alert(self):
        """Test publishing generic risk alert."""
        publisher = AlertPublisher(brokers=["localhost:9092"])
        publisher.connect()

        result = publisher.publish_risk_alert(
            alert_type=AlertType.POSITION_CONCENTRATION,
            symbol="NVDA",
            message="Position concentration exceeds 25% of portfolio",
            data={"concentration_pct": 0.28},
            priority="high",
        )

        assert result is True

    def test_build_event_structure(self):
        """Test that events have correct structure."""
        publisher = AlertPublisher(brokers=["localhost:9092"])

        event = publisher._build_event(
            alert_type=AlertType.STOP_LOSS_SET,
            symbol="AAPL",
            message="Test message",
            data={"test_key": "test_value"},
            metadata={"source": "test"},
            priority="normal",
        )

        assert event["event_type"] == "RISK_ALERT"
        assert event["source"] == "risk-engine"
        assert event["schema_version"] == "1.0"
        assert "timestamp" in event
        assert event["data"]["alert_type"] == "STOP_LOSS_SET"
        assert event["data"]["symbol"] == "AAPL"
        assert event["data"]["message"] == "Test message"
        assert event["data"]["test_key"] == "test_value"
        assert event["data"]["priority"] == "normal"
        assert event["metadata"]["source"] == "test"

    def test_alert_type_enum(self):
        """Test AlertType enum values."""
        assert AlertType.STOP_LOSS_SET.value == "STOP_LOSS_SET"
        assert AlertType.STOP_LOSS_TRIGGERED.value == "STOP_LOSS_TRIGGERED"
        assert AlertType.STOP_LOSS_APPROACHING.value == "STOP_LOSS_APPROACHING"
        assert AlertType.PROFIT_TARGET_APPROACHING.value == "PROFIT_TARGET_APPROACHING"
        assert AlertType.RISK_LIMIT_EXCEEDED.value == "RISK_LIMIT_EXCEEDED"
        assert AlertType.POSITION_CONCENTRATION.value == "POSITION_CONCENTRATION"

    def test_close(self):
        """Test closing publisher."""
        publisher = AlertPublisher(brokers=["localhost:9092"])
        publisher.connect()
        publisher.close()

        assert publisher._producer is None
