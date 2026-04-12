from __future__ import annotations


def build_layer1_payload(
    *,
    is_fake: bool,
    confidence: float,
    content_classification: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "result": "FAKE" if is_fake else "REAL",
        "confidence": round(max(0.0, min(100.0, confidence * 100.0)), 2),
        "heatmap": None,
        "content_classification": content_classification
        or {
            "content_type": "unknown",
            "confidence": 0.0,
            "raw_label": "unknown",
            "all_scores": {
                "real_human": 0.0,
                "ai_generated_human": 0.0,
                "painting": 0.0,
                "digital_art": 0.0,
                "cartoon": 0.0,
                "real_scene": 0.0,
                "ai_generated_scene": 0.0,
                "document": 0.0,
            },
        },
    }

