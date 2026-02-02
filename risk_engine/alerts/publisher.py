"""
Alert publisher for risk notifications.

Publishes stop loss and risk alerts to Kafka for consumption by alert-service.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import kafka, but make it optional
try:
    from kafka import KafkaProducer
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False
    KafkaProducer = None


class AlertType(str, Enum):
    """Types of risk alerts."""
    STOP_LOSS_SET = "STOP_LOSS_SET"
    STOP_LOSS_TRIGGERED = "STOP_LOSS_TRIGGERED"
    STOP_LOSS_APPROACHING = "STOP_LOSS_APPROACHING"
    PROFIT_TARGET_APPROACHING = "PROFIT_TARGET_APPROACHING"
    RISK_LIMIT_EXCEEDED = "RISK_LIMIT_EXCEEDED"
    POSITION_CONCENTRATION = "POSITION_CONCENTRATION"


class AlertPublisher:
    """
    Publishes risk alerts to Kafka.

    Alerts are consumed by alert-service and forwarded to Telegram.
    """

    def __init__(
        self,
        brokers: List[str],
        topic: str = "trading.risk-alerts",
    ):
        """
        Initialize the alert publisher.

        Args:
            brokers: List of Kafka broker addresses
            topic: Kafka topic for risk alerts
        """
        self.brokers = brokers
        self.topic = topic
        self._producer: Optional[KafkaProducer] = None

    def connect(self) -> bool:
        """Connect to Kafka."""
        if not HAS_KAFKA:
            logger.warning("kafka-python not installed, alerts will be logged only")
            return True

        try:
            self._producer = KafkaProducer(
                bootstrap_servers=self.brokers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
            )
            logger.info(f"Alert publisher connected to Kafka: {self.brokers}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Kafka for alerts: {e}")
            return False

    def close(self) -> None:
        """Close Kafka connection."""
        if self._producer:
            self._producer.close()
            self._producer = None

    def publish_stop_loss_alert(
        self,
        symbol: str,
        shares: float,
        average_cost: float,
        stop_loss_price: float,
        stop_loss_pct: float,
        current_price: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Publish a stop loss setup alert.

        This is sent when a position is opened to remind the user
        to set their stop loss.

        Args:
            symbol: Stock symbol
            shares: Number of shares
            average_cost: Average cost per share
            stop_loss_price: Calculated stop loss price
            stop_loss_pct: Stop loss percentage
            current_price: Current market price
            metadata: Optional additional data

        Returns:
            True if published successfully
        """
        distance_to_stop = current_price - stop_loss_price
        distance_pct = distance_to_stop / current_price if current_price > 0 else 0

        event = self._build_event(
            alert_type=AlertType.STOP_LOSS_SET,
            symbol=symbol,
            message=(
                f"Set stop loss for {symbol}\n"
                f"Shares: {shares:.0f}\n"
                f"Avg Cost: ${average_cost:.2f}\n"
                f"Stop Price: ${stop_loss_price:.2f} ({stop_loss_pct*100:.1f}% below avg)\n"
                f"Current: ${current_price:.2f}\n"
                f"Distance: ${distance_to_stop:.2f} ({distance_pct*100:.1f}%)"
            ),
            data={
                "shares": shares,
                "average_cost": average_cost,
                "stop_loss_price": stop_loss_price,
                "stop_loss_pct": stop_loss_pct,
                "current_price": current_price,
                "distance_to_stop": distance_to_stop,
                "distance_to_stop_pct": distance_pct,
            },
            metadata=metadata,
        )

        return self._publish(symbol, event)

    def publish_stop_triggered_alert(
        self,
        symbol: str,
        shares: float,
        average_cost: float,
        stop_loss_price: float,
        current_price: float,
        loss_amount: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Publish alert when stop loss has been triggered.

        Args:
            symbol: Stock symbol
            shares: Number of shares
            average_cost: Average cost per share
            stop_loss_price: Stop loss price
            current_price: Current market price
            loss_amount: Total dollar loss
            metadata: Optional additional data

        Returns:
            True if published successfully
        """
        event = self._build_event(
            alert_type=AlertType.STOP_LOSS_TRIGGERED,
            symbol=symbol,
            message=(
                f"⚠️ STOP LOSS TRIGGERED: {symbol}\n"
                f"Current: ${current_price:.2f}\n"
                f"Stop: ${stop_loss_price:.2f}\n"
                f"Avg Cost: ${average_cost:.2f}\n"
                f"Loss: ${loss_amount:.2f}"
            ),
            data={
                "shares": shares,
                "average_cost": average_cost,
                "stop_loss_price": stop_loss_price,
                "current_price": current_price,
                "loss_amount": loss_amount,
            },
            metadata=metadata,
            priority="high",
        )

        return self._publish(symbol, event)

    def publish_stop_approaching_alert(
        self,
        symbol: str,
        current_price: float,
        stop_loss_price: float,
        distance_pct: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Publish alert when price is approaching stop loss.

        Args:
            symbol: Stock symbol
            current_price: Current market price
            stop_loss_price: Stop loss price
            distance_pct: Percentage distance to stop
            metadata: Optional additional data

        Returns:
            True if published successfully
        """
        event = self._build_event(
            alert_type=AlertType.STOP_LOSS_APPROACHING,
            symbol=symbol,
            message=(
                f"⚡ {symbol} approaching stop loss\n"
                f"Current: ${current_price:.2f}\n"
                f"Stop: ${stop_loss_price:.2f}\n"
                f"Distance: {distance_pct*100:.1f}%"
            ),
            data={
                "current_price": current_price,
                "stop_loss_price": stop_loss_price,
                "distance_pct": distance_pct,
            },
            metadata=metadata,
            priority="medium",
        )

        return self._publish(symbol, event)

    def publish_risk_alert(
        self,
        alert_type: AlertType,
        symbol: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: str = "normal",
    ) -> bool:
        """
        Publish a generic risk alert.

        Args:
            alert_type: Type of alert
            symbol: Stock symbol
            message: Human-readable message
            data: Alert data
            metadata: Optional metadata
            priority: Alert priority (low, normal, medium, high)

        Returns:
            True if published successfully
        """
        event = self._build_event(
            alert_type=alert_type,
            symbol=symbol,
            message=message,
            data=data or {},
            metadata=metadata,
            priority=priority,
        )

        return self._publish(symbol, event)

    def _build_event(
        self,
        alert_type: AlertType,
        symbol: str,
        message: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        priority: str = "normal",
    ) -> Dict[str, Any]:
        """Build the alert event structure."""
        return {
            "event_type": "RISK_ALERT",
            "source": "risk-engine",
            "schema_version": "1.0",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": {
                "alert_type": alert_type.value,
                "symbol": symbol,
                "message": message,
                "priority": priority,
                **data,
            },
            "metadata": metadata or {},
        }

    def _publish(self, key: str, event: Dict[str, Any]) -> bool:
        """Publish event to Kafka."""
        # Always log the alert
        alert_type = event["data"].get("alert_type", "UNKNOWN")
        symbol = event["data"].get("symbol", "UNKNOWN")
        logger.info(f"Risk Alert [{alert_type}] {symbol}: {event['data'].get('message', '')}")

        if not self._producer:
            # Kafka not available, just log
            return True

        try:
            future = self._producer.send(self.topic, key=key, value=event)
            future.get(timeout=10)
            logger.debug(f"Published alert to {self.topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish alert: {e}")
            return False
