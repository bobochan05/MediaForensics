from __future__ import annotations

from backend.schemas.request_schema import AnalyzeRequest


def parse_analyze_request(payload: dict) -> AnalyzeRequest:
    return AnalyzeRequest.from_payload(payload)

