from __future__ import annotations

from datetime import datetime, timezone


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

