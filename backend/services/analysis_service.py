from __future__ import annotations


def build_layer1_payload(*, is_fake: bool, confidence: float) -> dict[str, object]:
    return {
        "result": "FAKE" if is_fake else "REAL",
        "confidence": round(max(0.0, min(100.0, confidence * 100.0)), 2),
        "heatmap": None,
    }

