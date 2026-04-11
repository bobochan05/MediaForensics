from __future__ import annotations

from ai.layer1_detection.utils import LABEL_TO_NAME
from ai.shared.preprocessing import media_type_from_path


def test_layer1_label_map():
    assert LABEL_TO_NAME[0] == "real"
    assert LABEL_TO_NAME[1] == "fake"


def test_media_type_resolution():
    assert media_type_from_path("sample.mp4") == "video"
    assert media_type_from_path("sample.png") == "image"

