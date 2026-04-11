from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from ai.layer2_matching.credibility import source_credibility_score
from ai.layer1_detection.frame_extractor import extract_sampled_frames
from ai.layer2_matching.tracking.metadata_parser import normalize_timestamp


LOGGER = logging.getLogger(__name__)

NEWS_PLATFORMS = {"news", "government"}
NEWS_DOMAINS = (
    "reuters",
    "apnews",
    "bbc",
    "nytimes",
    "cnn",
    "guardian",
    "theguardian",
    "washingtonpost",
    "wsj",
    "forbes",
    "bloomberg",
    "npr",
    "abcnews",
    "cbsnews",
    "nbcnews",
    "aljazeera",
    "ndtv",
    "indiatoday",
    "hindustantimes",
    "thehindu",
)
VIDEO_PLATFORMS = {"youtube", "video", "vimeo", "dailymotion", "twitch", "rumble", "bitchute"}
VIDEO_DOMAINS = ("youtube", "youtu.be", "vimeo", "dailymotion", "twitch", "rumble", "bitchute")
SOCIAL_PLATFORMS = {"reddit", "twitter", "instagram", "tiktok", "social", "facebook", "threads", "linkedin", "telegram", "discord", "snapchat", "pinterest", "tumblr"}
SOCIAL_DOMAINS = ("reddit", "x.com", "twitter", "instagram", "tiktok", "facebook", "fb.com", "threads.net", "linkedin", "telegram", "t.me", "discord", "snapchat", "pinterest", "tumblr")
OTHER_PLATFORMS = {"blog", "medium", "substack", "forum", "website", "web"}
OTHER_DOMAINS = ("medium", "substack", "wordpress", "blogspot", "quora", "fandom", "wikipedia", "tumblr", "pinterest")


def _field(item, name: str, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _tokenize(text: str | None) -> set[str]:
    lowered = str(text or "").lower()
    return {token for token in re.findall(r"[a-z0-9]+", lowered) if len(token) > 2}


def _candidate_locator(item) -> str:
    if _field(item, "url", None):
        return str(_field(item, "url"))
    if _field(item, "local_path", None):
        return str(_field(item, "local_path"))
    return str(_field(item, "id", ""))


def _candidate_media_path(item) -> Path | None:
    metadata = dict(_field(item, "metadata", {}) or {})
    for candidate in (metadata.get("downloaded_path"), metadata.get("candidate_path"), _field(item, "local_path", None)):
        text = str(candidate or "").strip()
        if text:
            path = Path(text)
            if path.exists():
                return path
    return None


def _infer_platform_bucket(url: str | None, platform: str | None) -> str:
    platform_name = str(platform or "").lower()
    domain = urlparse(str(url or "")).netloc.lower().removeprefix("www.")
    if platform_name in NEWS_PLATFORMS or any(name in domain for name in NEWS_DOMAINS):
        return "news"
    if platform_name in VIDEO_PLATFORMS or any(name in domain for name in VIDEO_DOMAINS):
        return "video"
    if platform_name in SOCIAL_PLATFORMS or any(name in domain for name in SOCIAL_DOMAINS):
        return "social"
    if platform_name in OTHER_PLATFORMS or any(name in domain for name in OTHER_DOMAINS):
        return "other"
    return "other"


def _build_match_reason(item, source_credibility: float, reference_tokens: set[str]) -> list[str]:
    reasons: list[str] = []
    metadata = dict(_field(item, "metadata", {}) or {})
    visual_similarity = _safe_float(_field(item, "visual_similarity", None), 0.0)
    if visual_similarity >= 0.85:
        reasons.append("high_visual_similarity")
    elif visual_similarity >= 0.65:
        reasons.append("moderate_visual_similarity")

    phash_diff = metadata.get("phash_diff")
    if isinstance(phash_diff, (int, float)) and int(phash_diff) <= 12:
        reasons.append("phash_match")

    title_tokens = _tokenize(_field(item, "title", None)) | _tokenize(_field(item, "caption", None))
    if reference_tokens and len(reference_tokens & title_tokens) >= 2:
        reasons.append("title_relevance")

    if source_credibility >= 0.8:
        reasons.append("reliable_source")
    elif source_credibility <= 0.4:
        reasons.append("low_credibility_source")

    if bool(metadata.get("exact_matches_hint")):
        reasons.append("provider_exact_match_hint")

    if not reasons:
        reasons.append("score_based_selection")
    return reasons


def _load_first_frame(media_path: str | Path) -> np.ndarray | None:
    try:
        frames = extract_sampled_frames(
            video_path=media_path,
            image_size=224,
            sample_fps=0.5,
            frames_per_video=1,
        )
    except Exception:
        return None
    if not frames:
        return None
    return np.asarray(frames[0].convert("RGB"), dtype=np.float32)


def _mutation_type(original_media_path: str | Path, candidate_media_path: str | Path | None) -> str:
    if candidate_media_path is None:
        return "none"

    original_frame = _load_first_frame(original_media_path)
    candidate_frame = _load_first_frame(candidate_media_path)
    if original_frame is None or candidate_frame is None:
        return "none"

    original_h, original_w = original_frame.shape[:2]
    candidate_h, candidate_w = candidate_frame.shape[:2]
    if min(original_h, original_w, candidate_h, candidate_w) <= 0:
        return "none"

    original_ratio = original_w / original_h
    candidate_ratio = candidate_w / candidate_h
    ratio_delta = abs(original_ratio - candidate_ratio)

    if ratio_delta > 0.08:
        return "crop"

    size_ratio = max(candidate_w / original_w, candidate_h / original_h)
    if abs(size_ratio - 1.0) > 0.18:
        return "resized"

    original_mean = float(original_frame.mean())
    candidate_mean = float(candidate_frame.mean())
    original_std = float(original_frame.std())
    candidate_std = float(candidate_frame.std())
    if abs(original_mean - candidate_mean) > 18.0 or abs(original_std - candidate_std) > 12.0:
        return "color_adjusted"

    return "none"


def _platform_distribution(items) -> dict[str, int]:
    counts = Counter({"news": 0, "social": 0, "video": 0, "other": 0})
    for item in items:
        counts[_infer_platform_bucket(_field(item, "url", None), _field(item, "platform", None))] += 1
    return dict(counts)


def _temporal_anomaly(items) -> bool:
    dated = []
    for item in items:
        normalized = normalize_timestamp(_field(item, "timestamp", None))
        if not normalized:
            continue
        source_cred = source_credibility_score(_field(item, "url", None), _field(item, "platform", None))
        dated.append((normalized, source_cred))

    if not dated:
        return False

    low = min((timestamp for timestamp, score in dated if score <= 0.4), default=None)
    high = min((timestamp for timestamp, score in dated if score >= 0.8), default=None)
    return bool(low and high and low < high)


def _cross_modal_consistency(items, audio_embedding_dim: int | None) -> str:
    if audio_embedding_dim is None:
        return "unknown"

    paired = [
        (float(_field(item, "visual_similarity")), float(_field(item, "audio_similarity")))
        for item in items
        if _field(item, "visual_similarity", None) is not None and _field(item, "audio_similarity", None) is not None
    ]
    if not paired:
        return "unknown"

    visual_mean = float(np.mean([visual for visual, _ in paired]))
    audio_mean = float(np.mean([audio for _, audio in paired]))
    return "consistent" if abs(visual_mean - audio_mean) <= 0.2 else "mismatch"


def build_layer2_insights(
    *,
    similar_content,
    original_media_path: str | Path,
    original_filename: str | None = None,
    query_hint: str | None = None,
    audio_embedding_dim: int | None = None,
) -> dict[str, Any]:
    insights: dict[str, Any] = {
        "platform_distribution": {"news": 0, "social": 0, "video": 0, "other": 0},
        "temporal_anomaly": False,
        "cross_modal_consistency": "unknown",
        "per_candidate_enrichment": [],
    }

    reference_tokens = _tokenize(original_filename) | _tokenize(query_hint)

    try:
        insights["platform_distribution"] = _platform_distribution(similar_content)
    except Exception:
        LOGGER.exception("Layer 2 insights: platform distribution failed")

    try:
        insights["temporal_anomaly"] = _temporal_anomaly(similar_content)
    except Exception:
        LOGGER.exception("Layer 2 insights: temporal anomaly detection failed")

    try:
        insights["cross_modal_consistency"] = _cross_modal_consistency(similar_content, audio_embedding_dim)
    except Exception:
        LOGGER.exception("Layer 2 insights: cross-modal consistency failed")

    enrichments: list[dict[str, Any]] = []
    for item in similar_content:
        try:
            source_credibility = source_credibility_score(getattr(item, "url", None), getattr(item, "platform", None))
        except Exception:
            LOGGER.exception("Layer 2 insights: source credibility failed for %s", _candidate_locator(item))
            source_credibility = 0.5

        try:
            match_reason = _build_match_reason(item, source_credibility, reference_tokens)
        except Exception:
            LOGGER.exception("Layer 2 insights: match reason failed for %s", _candidate_locator(item))
            match_reason = ["score_based_selection"]

        try:
            mutation_type = _mutation_type(original_media_path, _candidate_media_path(item))
        except Exception:
            LOGGER.exception("Layer 2 insights: mutation detection failed for %s", _candidate_locator(item))
            mutation_type = "none"

        enrichments.append(
            {
                "url": _candidate_locator(item),
                "source_credibility": round(float(source_credibility), 3),
                "match_reason": match_reason,
                "mutation_type": mutation_type,
            }
        )

    insights["per_candidate_enrichment"] = enrichments
    return insights
