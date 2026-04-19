from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
except ImportError:  # pragma: no cover - dependency availability varies by environment
    SendGridAPIClient = None  # type: ignore[assignment]
    Mail = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

LOGGER = logging.getLogger("tracelyt.email_alerts")
_ALERT_CACHE_LOCK = Lock()
_ALERT_CACHE: dict[str, datetime] = {}
_ALERT_COOLDOWN = timedelta(minutes=max(1, int(os.getenv("TRACELYT_ALERT_COOLDOWN_MINUTES", "30"))))


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _cleanup_cache(now: datetime) -> None:
    expired = [key for key, sent_at in _ALERT_CACHE.items() if (now - sent_at) >= _ALERT_COOLDOWN]
    for key in expired:
        _ALERT_CACHE.pop(key, None)


def _normalize_detection(value: object) -> str:
    cleaned = str(value or "").strip().upper()
    if cleaned in {"FAKE", "AI_GENERATED", "SYNTHETIC"}:
        return "AI_GENERATED"
    if cleaned in {"REAL", "AUTHENTIC"}:
        return "REAL"
    return cleaned or "UNKNOWN"


def _normalize_fraction(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric > 1.0:
        numeric = numeric / 100.0
    return max(0.0, min(1.0, numeric))


def _normalize_risk_score(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 <= numeric <= 1.0:
        numeric *= 10.0
    return max(0.0, min(10.0, numeric))


def _normalize_velocity(value: object) -> str:
    if isinstance(value, (int, float)):
        return "Increasing" if float(value) > 0.35 else "Stable"
    cleaned = str(value or "").strip().lower()
    if cleaned in {"increasing", "rapid", "rising", "growing"}:
        return "Increasing"
    if cleaned in {"declining", "decreasing"}:
        return "Declining"
    if cleaned in {"stable", "flat"}:
        return "Stable"
    return cleaned.title() or "Unknown"


def _normalize_signals(value: object) -> list[str]:
    if isinstance(value, dict):
        signals: list[str] = []
        for key, signal_value in value.items():
            label = str(key or "").strip().replace("_", " ")
            detail = str(signal_value or "").strip()
            if label and detail:
                signals.append(f"{label.title()}: {detail}")
            elif label:
                signals.append(label.title())
        return signals
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _fingerprint_payload(to_email: str, result: dict[str, Any]) -> str:
    stable_payload = {
        "to_email": to_email.strip().lower(),
        "content_id": str(result.get("content_id") or "").strip(),
        "cluster_id": str(result.get("cluster_id") or "").strip(),
        "upload_id": str(result.get("upload_id") or "").strip(),
        "detection": _normalize_detection(result.get("detection")),
        "confidence": round(_normalize_fraction(result.get("confidence")), 4),
        "risk_score": round(_normalize_risk_score(result.get("risk_score")), 2),
        "velocity": _normalize_velocity(result.get("velocity")),
        "signals": _normalize_signals(result.get("signals"))[:6],
    }
    return hashlib.sha1(json.dumps(stable_payload, sort_keys=True).encode("utf-8")).hexdigest()


def _email_html(result: dict[str, Any]) -> str:
    detection = _normalize_detection(result.get("detection"))
    confidence_fraction = _normalize_fraction(result.get("confidence"))
    risk_score = _normalize_risk_score(result.get("risk_score"))
    spread_status = str(result.get("spread_status") or _normalize_velocity(result.get("velocity"))).strip() or "Unknown"
    timestamp = str(result.get("timestamp") or _utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
    signals = _normalize_signals(result.get("signals")) or ["No forensic signals were supplied in this alert payload."]
    signal_items = "".join(
        f"<li style=\"margin:0 0 8px 0;color:#dbe7f5;line-height:1.5;\">{signal}</li>" for signal in signals[:6]
    )

    return f"""
    <div style="background:#07111f;padding:32px 16px;font-family:Arial,Helvetica,sans-serif;color:#e5eef8;">
      <div style="max-width:680px;margin:0 auto;background:#0d1726;border:1px solid #1f3148;border-radius:18px;overflow:hidden;box-shadow:0 16px 32px rgba(0,0,0,0.28);">
        <div style="padding:24px 28px;border-bottom:1px solid #1f3148;background:#0a1422;">
          <div style="font-size:11px;letter-spacing:0.14em;text-transform:uppercase;color:#8aa0ba;font-weight:700;">Tracelyt Intelligence System</div>
          <h1 style="margin:10px 0 0 0;font-size:24px;line-height:1.2;color:#f8fbff;">High Risk Synthetic Content Detected</h1>
        </div>
        <div style="padding:24px 28px;">
          <div style="display:block;margin-bottom:18px;padding:16px 18px;background:#101d2f;border:1px solid #213754;border-radius:14px;">
            <div style="font-size:12px;color:#93a7be;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;">Alert Summary</div>
            <table style="width:100%;border-collapse:collapse;">
              <tr>
                <td style="padding:6px 0;color:#93a7be;width:44%;">Detection</td>
                <td style="padding:6px 0;color:#f4f8fc;font-weight:700;">{detection}</td>
              </tr>
              <tr>
                <td style="padding:6px 0;color:#93a7be;">Confidence Score</td>
                <td style="padding:6px 0;color:#f4f8fc;font-weight:700;">{confidence_fraction:.2f} ({confidence_fraction * 100:.1f}%)</td>
              </tr>
              <tr>
                <td style="padding:6px 0;color:#93a7be;">Risk Score</td>
                <td style="padding:6px 0;color:#f4f8fc;font-weight:700;">{risk_score:.1f} / 10</td>
              </tr>
              <tr>
                <td style="padding:6px 0;color:#93a7be;">Spread Status</td>
                <td style="padding:6px 0;color:#f4f8fc;font-weight:700;">{spread_status}</td>
              </tr>
              <tr>
                <td style="padding:6px 0;color:#93a7be;">Timestamp</td>
                <td style="padding:6px 0;color:#f4f8fc;font-weight:700;">{timestamp}</td>
              </tr>
            </table>
          </div>
          <div style="padding:16px 18px;background:#0b1524;border:1px solid #1f3148;border-radius:14px;">
            <div style="font-size:12px;color:#93a7be;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;">Key Forensic Signals</div>
            <ul style="margin:0;padding-left:18px;">
              {signal_items}
            </ul>
          </div>
        </div>
        <div style="padding:16px 28px;border-top:1px solid #1f3148;background:#0a1422;font-size:12px;color:#8aa0ba;">
          Tracelyt Intelligence System • Automated Layer 3 tracking alert
        </div>
      </div>
    </div>
    """


def send_alert_email(to_email: str, result: dict[str, Any]) -> dict[str, Any]:
    api_key = str(os.getenv("SENDGRID_API_KEY") or "").strip()
    sender_email = str(os.getenv("SENDER_EMAIL") or "").strip()
    recipient = str(to_email or "").strip()

    if not api_key:
        raise RuntimeError("Missing SENDGRID_API_KEY environment variable.")
    if not sender_email:
        raise RuntimeError("Missing SENDER_EMAIL environment variable.")
    if not recipient:
        raise ValueError("A recipient email address is required.")
    if SendGridAPIClient is None or Mail is None:
        raise RuntimeError("SendGrid SDK is not installed. Add 'sendgrid' to the environment before sending alerts.")

    message = Mail(
        from_email=sender_email,
        to_emails=recipient,
        subject="🚨 Tracelyt Alert: High Risk Content Detected",
        html_content=_email_html(result),
    )

    try:
        client = SendGridAPIClient(api_key)
        response = client.send(message)
        body = response.body.decode("utf-8", errors="ignore") if isinstance(response.body, bytes) else str(response.body or "")
        print(f"SendGrid status code: {response.status_code}")
        if body:
            print(f"SendGrid response body: {body}")
        if response.status_code >= 400:
            if "verified" in body.lower():
                LOGGER.error("SendGrid sender verification issue detected for sender '%s'.", sender_email)
            raise RuntimeError(f"SendGrid email delivery failed with status {response.status_code}.")
        return {
            "ok": response.status_code == 202,
            "status_code": response.status_code,
            "body": body,
            "to_email": recipient,
        }
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        body = getattr(exc, "body", b"")
        body_text = body.decode("utf-8", errors="ignore") if isinstance(body, bytes) else str(body or "")
        if status_code is not None:
            print(f"SendGrid status code: {status_code}")
        if body_text:
            print(f"SendGrid response body: {body_text}")
            if "verified" in body_text.lower():
                LOGGER.error("SendGrid sender verification issue detected for sender '%s'.", sender_email)
        LOGGER.exception("Tracelyt alert email delivery failed for %s", recipient)
        raise


def handle_layer3_result(result: dict[str, Any], user_email: str) -> dict[str, Any]:
    detection = _normalize_detection(result.get("detection"))
    risk_score = _normalize_risk_score(result.get("risk_score"))
    velocity = _normalize_velocity(result.get("velocity"))

    if detection != "AI_GENERATED":
        return {"sent": False, "reason": "Detection did not indicate synthetic media."}
    if risk_score < 7.0 and velocity != "Increasing":
        return {"sent": False, "reason": "Layer 3 alert threshold not met."}

    now = _utcnow()
    fingerprint = _fingerprint_payload(user_email, result)
    with _ALERT_CACHE_LOCK:
        _cleanup_cache(now)
        last_sent_at = _ALERT_CACHE.get(fingerprint)
        if last_sent_at and (now - last_sent_at) < _ALERT_COOLDOWN:
            return {
                "sent": False,
                "reason": "Alert suppressed by cooldown.",
                "cooldown_until": (last_sent_at + _ALERT_COOLDOWN).isoformat(),
            }

    delivery = send_alert_email(user_email, result)
    with _ALERT_CACHE_LOCK:
        _ALERT_CACHE[fingerprint] = now
    return {
        "sent": bool(delivery.get("ok")),
        "reason": "Alert delivered." if delivery.get("ok") else "Alert attempted.",
        "fingerprint": fingerprint,
        "delivery": delivery,
    }
