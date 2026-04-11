from __future__ import annotations

from typing import Any


def json_error(message: str, *, code: str = "error", **extra: Any) -> dict[str, Any]:
    return {"status": "error", "code": code, "error": message, **extra}


def json_ok(data: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "ok"}
    if data is not None:
        payload["data"] = data
    payload.update(extra)
    return payload

