from __future__ import annotations

from backend.services.analysis_service import build_layer1_payload
from ai.layer1_detection.utils import LABEL_TO_NAME
from ai.shared.preprocessing import media_type_from_path


def test_layer1_label_map():
    assert LABEL_TO_NAME[0] == "real"
    assert LABEL_TO_NAME[1] == "fake"


def test_media_type_resolution():
    assert media_type_from_path("sample.mp4") == "video"
    assert media_type_from_path("sample.png") == "image"


def test_build_layer1_payload_includes_content_classification():
    payload = build_layer1_payload(is_fake=True, confidence=0.91)
    assert payload["result"] == "FAKE"
    assert payload["confidence"] == 91.0
    assert "content_classification" in payload
    content = payload["content_classification"]
    assert content["content_type"] == "unknown"
    assert content["raw_label"] == "unknown"
    assert set(content["all_scores"].keys()) == {
        "real_human",
        "ai_generated_human",
        "painting",
        "digital_art",
        "cartoon",
        "real_scene",
        "ai_generated_scene",
        "document",
    }

