from __future__ import annotations

from datetime import datetime, timezone


def build_growth_indicator(growth_rate_percent: float) -> str:
    if growth_rate_percent >= 120:
        return "viral"
    if growth_rate_percent >= 40:
        return "medium"
    return "low"


def build_risk_insight(*, fake_probability: float, growth_rate_percent: float, source_count: int) -> str:
    if fake_probability >= 0.85 and growth_rate_percent >= 80:
        return "High risk: likely manipulated media with active spread."
    if fake_probability >= 0.7 or source_count >= 4:
        return "Moderate risk: suspicious media with measurable propagation."
    return "Low immediate risk: continue monitoring for spread changes."


def build_alerts(*, growth_rate_percent: float, source_count: int) -> list[dict[str, object]]:
    now_iso = datetime.now(timezone.utc).isoformat()
    alerts: list[dict[str, object]] = []
    if growth_rate_percent >= 120:
        alerts.append(
            {
                "severity": "warning",
                "title": "Spike detected",
                "message": f"Spread velocity increased by {round(growth_rate_percent, 2)}% in the active window.",
                "created_at": now_iso,
            }
        )
    if source_count >= 5:
        alerts.append(
            {
                "severity": "info",
                "title": "New source cluster",
                "message": "Multiple independent source references were detected in this run.",
                "created_at": now_iso,
            }
        )
    return alerts

