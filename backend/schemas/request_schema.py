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
    tracking_enabled: bool

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AnalyzeRequest":
        source_url = str(payload.get("source_url") or "").strip() or None
        return cls(
            source_url=source_url,
            enable_layer1=_as_bool(payload.get("enable_layer1"), True),
            enable_layer2=_as_bool(payload.get("enable_layer2"), True),
            enable_layer3=_as_bool(payload.get("enable_layer3"), True),
            tracking_enabled=_as_bool(payload.get("tracking_enabled"), False),
        )


@dataclass(frozen=True)
class ChatRequest:
    message: str
    analysis_id: str | None
    layer1: dict[str, Any]
    layer2: dict[str, Any]
    layer3: dict[str, Any]
    history: list[dict[str, str]]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ChatRequest":
        message = str(payload.get("message") or "").strip()
        if not message:
            raise ValueError("message is required.")
        analysis_id = str(payload.get("analysis_id") or "").strip() or None
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        history_payload = payload.get("history") if isinstance(payload.get("history"), list) else []
        history: list[dict[str, str]] = []
        for item in history_payload[-10:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or item.get("text") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            history.append({"role": role, "content": content[:1200]})
        return cls(
            message=message,
            analysis_id=analysis_id,
            layer1=context.get("layer1") if isinstance(context.get("layer1"), dict) else (payload.get("layer1") if isinstance(payload.get("layer1"), dict) else {}),
            layer2=context.get("layer2") if isinstance(context.get("layer2"), dict) else (payload.get("layer2") if isinstance(payload.get("layer2"), dict) else {}),
            layer3=context.get("layer3") if isinstance(context.get("layer3"), dict) else (payload.get("layer3") if isinstance(payload.get("layer3"), dict) else {}),
            history=history,
        )

