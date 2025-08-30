from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    labels: dict[str, str]
    value: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "severity": self.severity.value,
            "timestamp_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)),
        }


class AlertRule:
    """Define an alerting rule."""

    def __init__(
        self,
        name: str,
        condition: Callable[[float], bool],
        severity: AlertSeverity,
        message_template: str,
        threshold: float,
        cooldown_seconds: float = 300,  # 5 minutes
    ):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.message_template = message_template
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.last_fired = 0.0

    def check(self, value: float, labels: dict[str, str] = None) -> Alert | None:
        """Check if the rule should fire."""
        if not self.condition(value):
            return None

        # Check cooldown
        current_time = time.time()
        if current_time - self.last_fired < self.cooldown_seconds:
            return None

        self.last_fired = current_time

        return Alert(
            name=self.name,
            severity=self.severity,
            message=self.message_template.format(value=value, threshold=self.threshold),
            timestamp=current_time,
            labels=labels or {},
            value=value,
            threshold=self.threshold,
        )


class AlertManager:
    """Manage alerts and notifications."""

    def __init__(self):
        self.rules: list[AlertRule] = []
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.max_history_size = 1000

        # Default alert rules
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default alerting rules."""
        self.rules = [
            # GPU Memory alerts
            AlertRule(
                name="gpu_memory_high",
                condition=lambda x: x > 90,
                severity=AlertSeverity.WARNING,
                message_template="GPU memory usage is high: {value:.1f}% (threshold: {threshold}%)",
                threshold=90.0,
                cooldown_seconds=300,
            ),
            AlertRule(
                name="gpu_memory_critical",
                condition=lambda x: x > 95,
                severity=AlertSeverity.CRITICAL,
                message_template="GPU memory usage is critical: {value:.1f}% (threshold: {threshold}%)",
                threshold=95.0,
                cooldown_seconds=180,
            ),
            # Response time alerts
            AlertRule(
                name="response_time_high",
                condition=lambda x: x > 2.0,
                severity=AlertSeverity.WARNING,
                message_template="Average response time is high: {value:.2f}s (threshold: {threshold}s)",
                threshold=2.0,
                cooldown_seconds=300,
            ),
            AlertRule(
                name="response_time_critical",
                condition=lambda x: x > 5.0,
                severity=AlertSeverity.CRITICAL,
                message_template="Average response time is critical: {value:.2f}s (threshold: {threshold}s)",
                threshold=5.0,
                cooldown_seconds=180,
            ),
            # Error rate alerts
            AlertRule(
                name="error_rate_high",
                condition=lambda x: x > 5.0,
                severity=AlertSeverity.WARNING,
                message_template="Error rate is high: {value:.2f}% (threshold: {threshold}%)",
                threshold=5.0,
                cooldown_seconds=300,
            ),
            AlertRule(
                name="error_rate_critical",
                condition=lambda x: x > 10.0,
                severity=AlertSeverity.CRITICAL,
                message_template="Error rate is critical: {value:.2f}% (threshold: {threshold}%)",
                threshold=10.0,
                cooldown_seconds=180,
            ),
            # System resource alerts
            AlertRule(
                name="cpu_usage_high",
                condition=lambda x: x > 80,
                severity=AlertSeverity.WARNING,
                message_template="CPU usage is high: {value:.1f}% (threshold: {threshold}%)",
                threshold=80.0,
                cooldown_seconds=300,
            ),
            AlertRule(
                name="memory_usage_high",
                condition=lambda x: x > 85,
                severity=AlertSeverity.WARNING,
                message_template="Memory usage is high: {value:.1f}% (threshold: {threshold}%)",
                threshold=85.0,
                cooldown_seconds=300,
            ),
        ]

    def add_rule(self, rule: AlertRule):
        """Add a custom alerting rule."""
        self.rules.append(rule)

    def check_metric(
        self, metric_name: str, value: float, labels: dict[str, str] = None
    ) -> list[Alert]:
        """Check a metric value against all relevant rules."""
        fired_alerts = []

        for rule in self.rules:
            # Simple metric name matching - could be more sophisticated
            if metric_name in rule.name or rule.name in metric_name:
                alert = rule.check(value, labels)
                if alert:
                    fired_alerts.append(alert)
                    self._handle_alert(alert)

        return fired_alerts

    def _handle_alert(self, alert: Alert):
        """Handle a fired alert."""
        alert_key = f"{alert.name}:{hash(str(alert.labels))}"

        # Store active alert
        self.active_alerts[alert_key] = alert

        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size :]

        # Log the alert
        logger.warning(
            "alert_fired",
            name=alert.name,
            severity=alert.severity.value,
            message=alert.message,
            value=alert.value,
            threshold=alert.threshold,
            labels=alert.labels,
        )

        # Send notifications (webhook, email, etc.)
        asyncio.create_task(self._send_notifications(alert))

    async def _send_notifications(self, alert: Alert):
        """Send alert notifications."""
        try:
            # This could be extended to send to various notification channels
            # For now, just structured logging

            # Example: webhook notification
            # await self._send_webhook(alert)

            # Example: email notification
            # await self._send_email(alert)

            logger.info("alert_notification_sent", alert=alert.to_dict())

        except Exception as e:
            logger.exception("Failed to send alert notification", error=str(e))

    async def _send_webhook(self, alert: Alert, webhook_url: str = None):
        """Send alert to webhook (placeholder implementation)."""
        if not webhook_url:
            return

        # Implementation would use httpx or similar to POST to webhook
        # payload = {"alert": alert.to_dict(), "service": "vllm-inference-service"}

        # async with httpx.AsyncClient() as client:
        #     await client.post(webhook_url, json=payload)

    def get_active_alerts(self) -> list[dict[str, Any]]:
        """Get all active alerts."""
        return [alert.to_dict() for alert in self.active_alerts.values()]

    def get_alert_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent alert history."""
        recent_alerts = self.alert_history[-limit:] if limit else self.alert_history
        return [alert.to_dict() for alert in recent_alerts]

    def resolve_alert(self, alert_name: str, labels: dict[str, str] = None):
        """Manually resolve an alert."""
        alert_key = f"{alert_name}:{hash(str(labels or {}))}"
        if alert_key in self.active_alerts:
            resolved_alert = self.active_alerts.pop(alert_key)
            logger.info(
                "alert_resolved",
                name=resolved_alert.name,
                message="Alert manually resolved",
                labels=resolved_alert.labels,
            )


# Global alert manager instance
alert_manager = AlertManager()


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    return alert_manager


# Utility functions for common alert checks
def check_gpu_memory_alerts(gpu_index: str, memory_usage_percent: float):
    """Check GPU memory usage and fire alerts if necessary."""
    labels = {"gpu_index": gpu_index}
    alert_manager.check_metric("gpu_memory", memory_usage_percent, labels)


def check_response_time_alerts(avg_response_time: float):
    """Check response time and fire alerts if necessary."""
    alert_manager.check_metric("response_time", avg_response_time)


def check_error_rate_alerts(error_rate_percent: float):
    """Check error rate and fire alerts if necessary."""
    alert_manager.check_metric("error_rate", error_rate_percent)


def check_system_resource_alerts(cpu_percent: float, memory_percent: float):
    """Check system resource usage and fire alerts if necessary."""
    alert_manager.check_metric("cpu_usage", cpu_percent)
    alert_manager.check_metric("memory_usage", memory_percent)
