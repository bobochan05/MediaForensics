from __future__ import annotations

from backend.schemas.request_schema import ChatRequest


def parse_chat_request(payload: dict) -> ChatRequest:
    return ChatRequest.from_payload(payload)

