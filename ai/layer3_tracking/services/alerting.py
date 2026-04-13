from __future__ import annotations

import json
import logging
import os
import smtplib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

LOGGER = logging.getLogger("layer3.alerting")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class AlertEvent:
    event_type: str
    severity: str
    message: str
    content_id: str | None = None
    cluster_id: str | None = None
    error_type: str | None = None
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: _utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = str(self.severity or "LOW").upper()
        return payload


class Layer3AlertService:
    def __init__(self) -> None:
        self.root_dir = Path(__file__).resolve().parents[3] / "artifacts" / "layer3" / "alerts"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root_dir / "events.jsonl"
        self.failed_alerts_path = self.root_dir / "failed_alerts.jsonl"
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="layer3-alerts")
        self._lock = Lock()
        self._rate_limit: dict[str, datetime] = {}
        self._memory_feed: deque[dict[str, Any]] = deque(maxlen=80)
        self.rate_limit_window = timedelta(minutes=int(os.getenv("LAYER3_ALERT_WINDOW_MINUTES", "10")))
        self.smtp_host = str(os.getenv("LAYER3_SMTP_HOST") or "").strip()
        self.smtp_port = int(os.getenv("LAYER3_SMTP_PORT") or "587")
        self.smtp_user = str(os.getenv("LAYER3_SMTP_USER") or "").strip()
        self.smtp_password = str(os.getenv("LAYER3_SMTP_PASSWORD") or "").strip()
        self.smtp_from = str(os.getenv("LAYER3_ALERT_FROM") or self.smtp_user or "layer3@localhost").strip()
        self.smtp_to = [item.strip() for item in str(os.getenv("LAYER3_ALERT_TO") or "").split(",") if item.strip()]
        self.smtp_tls = str(os.getenv("LAYER3_SMTP_TLS") or "1").strip() != "0"

    def _rate_limit_key(self, event: AlertEvent) -> str:
        return "|".join(
            [
                event.event_type,
                event.severity,
                str(event.content_id or ""),
                str(event.cluster_id or ""),
            ]
        )

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _log_structured(self, event: AlertEvent) -> None:
        payload = event.to_dict()
        LOGGER.log(
            logging.CRITICAL if payload["severity"] == "CRITICAL" else logging.WARNING if payload["severity"] in {"HIGH", "MEDIUM"} else logging.INFO,
            json.dumps(
                {
                    "timestamp": payload["timestamp"],
                    "event_id": payload["event_id"],
                    "event_type": payload["event_type"],
                    "error_type": payload.get("error_type"),
                    "content_id": payload.get("content_id"),
                    "cluster_id": payload.get("cluster_id"),
                    "severity": payload["severity"],
                    "message": payload["message"],
                    "metadata": payload.get("metadata") or {},
                },
                ensure_ascii=True,
            ),
        )

    def _send_email(self, event: AlertEvent) -> None:
        if not self.smtp_host or not self.smtp_to:
            return
        message = EmailMessage()
        message["Subject"] = f"[Layer3][{event.severity}] {event.event_type}"
        message["From"] = self.smtp_from
        message["To"] = ", ".join(self.smtp_to)
        message.set_content(
            "\n".join(
                [
                    f"Event: {event.event_type}",
                    f"Severity: {event.severity}",
                    f"Content ID: {event.content_id or 'n/a'}",
                    f"Cluster ID: {event.cluster_id or 'n/a'}",
                    f"Timestamp: {event.timestamp}",
                    f"Message: {event.message}",
                    f"Explanation: {event.explanation or 'n/a'}",
                ]
            )
        )
        attempts = 3
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=12) as server:
                    if self.smtp_tls:
                        server.starttls()
                    if self.smtp_user:
                        server.login(self.smtp_user, self.smtp_password)
                    server.send_message(message)
                return
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
        failed_payload = event.to_dict()
        failed_payload["email_failure"] = str(last_error or "unknown email delivery error")
        self._append_jsonl(self.failed_alerts_path, failed_payload)
        self._log_structured(
            AlertEvent(
                event_type="email_delivery_failure",
                severity="CRITICAL",
                message="Layer 3 alert email delivery failed after retries.",
                content_id=event.content_id,
                cluster_id=event.cluster_id,
                error_type=type(last_error).__name__ if last_error else "EmailFailure",
                explanation=str(last_error or "unknown email delivery error"),
                metadata={"original_event_type": event.event_type},
            )
        )

    def _process(self, event: AlertEvent) -> None:
        key = self._rate_limit_key(event)
        now = _utcnow()
        with self._lock:
            previous = self._rate_limit.get(key)
            if previous and (now - previous) < self.rate_limit_window:
                return
            self._rate_limit[key] = now

        payload = event.to_dict()
        self._append_jsonl(self.events_path, payload)
        with self._lock:
            self._memory_feed.appendleft(payload)
        self._log_structured(event)
        if payload["severity"] in {"HIGH", "CRITICAL"}:
            self._send_email(event)

    def trigger_alert(self, event: AlertEvent) -> None:
        self._executor.submit(self._process, event)

    def recent_notifications(self, limit: int = 12) -> list[dict[str, Any]]:
        with self._lock:
            if self._memory_feed:
                return list(list(self._memory_feed)[: max(1, int(limit))])
        if not self.events_path.exists():
            return []
        lines = self.events_path.read_text(encoding="utf-8").splitlines()
        events: list[dict[str, Any]] = []
        for line in reversed(lines[-max(1, int(limit * 3)) :]):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(events) >= limit:
                break
        return events


_ALERT_SERVICE: Layer3AlertService | None = None
_ALERT_SERVICE_LOCK = Lock()


def get_alert_service() -> Layer3AlertService:
    global _ALERT_SERVICE
    with _ALERT_SERVICE_LOCK:
        if _ALERT_SERVICE is None:
            _ALERT_SERVICE = Layer3AlertService()
        return _ALERT_SERVICE

