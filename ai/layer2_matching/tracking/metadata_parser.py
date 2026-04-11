from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


PLATFORM_CREDIBILITY = {
    "government": 0.95,
    "news": 0.82,
    "youtube": 0.55,
    "reddit": 0.45,
    "twitter": 0.35,
    "blog": 0.4,
    "local_dataset": 0.7,
    "local_upload": 0.55,
    "external": 0.5,
    "unknown": 0.5,
}


@dataclass
class OccurrenceRecord:
    entry_id: str
    source_type: str
    platform: str
    url: str | None = None
    local_path: str | None = None
    timestamp: str | None = None
    title: str | None = None
    caption: str | None = None
    label: str | None = None
    credibility_score: float = 0.5
    visual_similarity: float | None = None
    audio_similarity: float | None = None
    fused_similarity: float = 0.0
    context: str = "news"
    context_scores: dict[str, float] = field(default_factory=dict)
    is_mock: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_timestamp(value: str | float | int | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    else:
        cleaned = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(cleaned)
        except ValueError:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def infer_platform(url: str | None = None, fallback: str | None = None) -> str:
    if not url:
        return fallback or "unknown"

    netloc = urlparse(url).netloc.lower()
    if "reddit" in netloc:
        return "reddit"
    if "twitter" in netloc or "x.com" in netloc:
        return "twitter"
    if "facebook" in netloc or "fb.com" in netloc:
        return "facebook"
    if "instagram" in netloc:
        return "instagram"
    if "tiktok" in netloc:
        return "tiktok"
    if "threads.net" in netloc:
        return "threads"
    if "linkedin" in netloc:
        return "linkedin"
    if "telegram" in netloc or "t.me" in netloc:
        return "telegram"
    if "discord" in netloc:
        return "discord"
    if "snapchat" in netloc:
        return "snapchat"
    if "pinterest" in netloc:
        return "pinterest"
    if "tumblr" in netloc:
        return "tumblr"
    if "youtube" in netloc or "youtu.be" in netloc:
        return "youtube"
    if "vimeo" in netloc:
        return "vimeo"
    if "dailymotion" in netloc:
        return "dailymotion"
    if "twitch" in netloc:
        return "twitch"
    if "rumble" in netloc:
        return "rumble"
    if "bitchute" in netloc:
        return "bitchute"
    if netloc.endswith(".gov"):
        return "government"
    if any(name in netloc for name in ("reuters", "apnews", "bbc", "nytimes", "cnn", "theguardian", "guardian", "washingtonpost", "forbes", "ndtv", "indiatoday", "hindustantimes")):
        return "news"
    if any(name in netloc for name in ("medium", "substack", "blog", "wordpress", "blogspot", "quora", "fandom")):
        return "blog"
    return fallback or "external"


def credibility_score_for_source(platform: str, url: str | None = None) -> float:
    platform = platform.lower()
    if url:
        netloc = urlparse(url).netloc.lower()
        if netloc.endswith(".gov"):
            return PLATFORM_CREDIBILITY["government"]
        if netloc.endswith(".edu"):
            return 0.9
        if any(name in netloc for name in ("reuters", "apnews", "bbc", "nytimes")):
            return 0.9
    return float(PLATFORM_CREDIBILITY.get(platform, PLATFORM_CREDIBILITY["unknown"]))


def timestamp_from_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    return normalize_timestamp(path_obj.stat().st_mtime)
