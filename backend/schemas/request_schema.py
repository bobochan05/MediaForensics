from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    cleaned = str(value).strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class AnalyzeRequest:
    source_url: str | None
    enable_layer1: bool
    enable_layer2: bool
    enable_layer3: bool

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AnalyzeRequest":
        source_url = str(payload.get("source_url") or "").strip() or None
        return cls(
            source_url=source_url,
            enable_layer1=_as_bool(payload.get("enable_layer1"), True),
            enable_layer2=_as_bool(payload.get("enable_layer2"), True),
            enable_layer3=_as_bool(payload.get("enable_layer3"), True),
        )


@dataclass(frozen=True)
class ChatRequest:
    message: str
    layer1: dict[str, Any]
    layer2: dict[str, Any]
    layer3: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ChatRequest":
        message = str(payload.get("message") or "").strip()
        if not message:
            raise ValueError("message is required.")
        return cls(
            message=message,
            layer1=payload.get("layer1") if isinstance(payload.get("layer1"), dict) else {},
            layer2=payload.get("layer2") if isinstance(payload.get("layer2"), dict) else {},
            layer3=payload.get("layer3") if isinstance(payload.get("layer3"), dict) else {},
        )

