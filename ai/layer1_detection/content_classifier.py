from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from ai.layer1_detection.frame_extractor import extract_sampled_frames
from ai.layer1_detection.models.clip_model import ClipModel

PROMPT_TO_CONTENT_TYPE: dict[str, str] = {
    "a real photo of a human face": "real_human",
    "an AI generated human face": "ai_generated_human",
    "a painting": "painting",
    "digital art": "digital_art",
    "a cartoon or anime character": "cartoon",
    "a real world scene or landscape": "real_scene",
    "an AI generated scene or landscape": "ai_generated_scene",
    "a document, screenshot, or text-based image": "document",
}

CONTENT_PROMPTS: list[str] = list(PROMPT_TO_CONTENT_TYPE.keys())
CONTENT_KEYS: list[str] = list(PROMPT_TO_CONTENT_TYPE.values())
UNKNOWN_CONFIDENCE_THRESHOLD = 0.16
UNKNOWN_MARGIN_THRESHOLD = 0.02
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
LOGGER = logging.getLogger(__name__)

_CONTENT_CLIP_MODEL: ClipModel | None = None
_CONTENT_CLIP_TEXT_FEATURES: np.ndarray | None = None
_CONTENT_CLIP_LOCK = Lock()


@dataclass(frozen=True)
class ContentClassification:
    content_type: str
    confidence: float
    raw_label: str
    all_scores: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "content_type": self.content_type,
            "confidence": round(float(self.confidence), 4),
            "raw_label": self.raw_label,
            "all_scores": {key: round(float(value), 4) for key, value in self.all_scores.items()},
        }


def _content_clip_model() -> tuple[ClipModel, np.ndarray]:
    global _CONTENT_CLIP_MODEL, _CONTENT_CLIP_TEXT_FEATURES
    with _CONTENT_CLIP_LOCK:
        if _CONTENT_CLIP_MODEL is None:
            _CONTENT_CLIP_MODEL = ClipModel(device="cpu")
        if _CONTENT_CLIP_TEXT_FEATURES is None:
            _CONTENT_CLIP_TEXT_FEATURES = _CONTENT_CLIP_MODEL.extract_text_features(CONTENT_PROMPTS)
    return _CONTENT_CLIP_MODEL, _CONTENT_CLIP_TEXT_FEATURES


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    denom = np.sum(exp_values)
    if float(denom) <= 0.0:
        return np.zeros_like(values, dtype=np.float32)
    return (exp_values / denom).astype(np.float32)


def _unknown_result() -> ContentClassification:
    return ContentClassification(
        content_type="unknown",
        confidence=0.0,
        raw_label="unknown",
        all_scores={key: 0.0 for key in CONTENT_KEYS},
    )


def _normalized(value: float, *, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(max(0.0, min(1.0, (value - low) / (high - low))))


def _heuristic_classification(frame) -> ContentClassification:
    rgb = np.asarray(frame.convert("RGB"))
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5, minSize=(36, 36))

    edge_density = float(np.mean(cv2.Canny(grayscale, 80, 160) > 0))
    laplacian_var = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
    brightness_std = float(np.std(grayscale) / 255.0)
    unique_colors = np.unique(rgb.reshape(-1, 3), axis=0).shape[0]
    unique_ratio = float(unique_colors / max(1, rgb.shape[0] * rgb.shape[1]))

    scores = {key: 0.02 for key in CONTENT_KEYS}

    if len(faces) > 0:
        smooth_skin = 1.0 - _normalized(laplacian_var, low=60.0, high=420.0)
        natural_face = _normalized(laplacian_var, low=80.0, high=520.0) * 0.55 + _normalized(brightness_std, low=0.06, high=0.24) * 0.45
        scores["real_human"] = max(scores["real_human"], min(0.98, 0.56 + natural_face * 0.38))
        scores["ai_generated_human"] = max(scores["ai_generated_human"], min(0.96, 0.36 + smooth_skin * 0.42 + max(0.0, saturation - 0.45) * 0.18))
    else:
        document_score = _normalized(edge_density, low=0.06, high=0.18) * 0.55 + (1.0 - saturation) * 0.45
        cartoon_score = _normalized(saturation, low=0.28, high=0.72) * 0.45 + (1.0 - unique_ratio) * 0.25 + (1.0 - _normalized(laplacian_var, low=80.0, high=360.0)) * 0.30
        painting_score = _normalized(saturation, low=0.22, high=0.64) * 0.30 + _normalized(edge_density, low=0.03, high=0.14) * 0.35 + (1.0 - _normalized(unique_ratio, low=0.22, high=0.9)) * 0.35
        digital_art_score = _normalized(saturation, low=0.35, high=0.82) * 0.45 + (1.0 - _normalized(edge_density, low=0.06, high=0.2)) * 0.20 + (1.0 - _normalized(laplacian_var, low=100.0, high=500.0)) * 0.35
        real_scene_score = _normalized(brightness_std, low=0.08, high=0.28) * 0.35 + _normalized(laplacian_var, low=90.0, high=460.0) * 0.40 + _normalized(unique_ratio, low=0.18, high=0.86) * 0.25
        ai_scene_score = (1.0 - _normalized(laplacian_var, low=90.0, high=420.0)) * 0.42 + _normalized(saturation, low=0.34, high=0.8) * 0.28 + (1.0 - _normalized(unique_ratio, low=0.18, high=0.85)) * 0.30

        scores["document"] = max(scores["document"], min(0.96, 0.26 + document_score * 0.62))
        scores["cartoon"] = max(scores["cartoon"], min(0.96, 0.24 + cartoon_score * 0.62))
        scores["painting"] = max(scores["painting"], min(0.95, 0.22 + painting_score * 0.60))
        scores["digital_art"] = max(scores["digital_art"], min(0.96, 0.24 + digital_art_score * 0.62))
        scores["real_scene"] = max(scores["real_scene"], min(0.96, 0.24 + real_scene_score * 0.62))
        scores["ai_generated_scene"] = max(scores["ai_generated_scene"], min(0.95, 0.24 + ai_scene_score * 0.60))

    total = float(sum(scores.values()))
    if total <= 0.0:
        return _unknown_result()
    normalized_scores = {key: float(value / total) for key, value in scores.items()}
    best_key = max(normalized_scores, key=normalized_scores.get)
    best_confidence = float(normalized_scores[best_key])
    if best_confidence < 0.18:
        best_key = "unknown"
    return ContentClassification(
        content_type=best_key,
        confidence=best_confidence,
        raw_label=best_key,
        all_scores=normalized_scores,
    )


def classify_media_content(media_path: str | Path) -> dict[str, object]:
    input_path = Path(media_path)
    if not input_path.exists():
        return _unknown_result().to_dict()

    try:
        frames = extract_sampled_frames(
            video_path=input_path,
            image_size=224,
            sample_fps=1.0,
            frames_per_video=3,
        )
        if not frames:
            return _unknown_result().to_dict()

        clip_model, text_features = _content_clip_model()
        image_features = clip_model.extract_features(frames)
        similarities = image_features @ text_features.T
        averaged_scores = np.mean(similarities, axis=0).astype(np.float32)
        probabilities = _softmax(averaged_scores)

        best_index = int(np.argmax(probabilities))
        best_prompt = CONTENT_PROMPTS[best_index]
        best_type = PROMPT_TO_CONTENT_TYPE[best_prompt]
        best_confidence = float(probabilities[best_index])

        sorted_probs = np.sort(probabilities)[::-1]
        second_confidence = float(sorted_probs[1]) if sorted_probs.size > 1 else 0.0
        if best_confidence < UNKNOWN_CONFIDENCE_THRESHOLD or (best_confidence - second_confidence) < UNKNOWN_MARGIN_THRESHOLD:
            best_type = "unknown"

        all_scores = {
            PROMPT_TO_CONTENT_TYPE[prompt]: float(probabilities[index])
            for index, prompt in enumerate(CONTENT_PROMPTS)
        }
        return ContentClassification(
            content_type=best_type,
            confidence=best_confidence,
            raw_label=best_prompt,
            all_scores=all_scores,
        ).to_dict()
    except Exception as exc:
        LOGGER.warning("Layer1 content classifier fell back to heuristics for %s: %s", input_path, exc)
        try:
            fallback = _heuristic_classification(frames[0] if 'frames' in locals() and frames else extract_sampled_frames(video_path=input_path, image_size=224, sample_fps=1.0, frames_per_video=1)[0])
            return fallback.to_dict()
        except Exception as fallback_exc:
            LOGGER.warning("Layer1 content classifier fallback failed for %s: %s", input_path, fallback_exc)
            return _unknown_result().to_dict()
