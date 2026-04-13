from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_PROFILE_CACHE: dict[tuple[str, int], "MediaProfile"] = {}
_PROFILE_CACHE_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class MediaProfile:
    path: str
    is_video: bool
    duration_seconds: float
    total_frames: int
    native_fps: float
    width: int
    height: int

    @property
    def is_long_video(self) -> bool:
        return self.is_video and self.duration_seconds >= 300.0

    @property
    def is_medium_video(self) -> bool:
        return self.is_video and 60.0 <= self.duration_seconds < 300.0

    @property
    def is_short_video(self) -> bool:
        return self.is_video and self.duration_seconds < 60.0


def profile_media(media_path: str | Path) -> MediaProfile:
    path = Path(media_path).resolve()
    try:
        signature = int(path.stat().st_mtime_ns)
    except FileNotFoundError:
        return MediaProfile(
            path=str(path),
            is_video=path.suffix.lower() in VIDEO_EXTENSIONS,
            duration_seconds=0.0,
            total_frames=0,
            native_fps=0.0,
            width=0,
            height=0,
        )

    cache_key = (str(path), signature)
    with _PROFILE_CACHE_LOCK:
        cached = _PROFILE_CACHE.get(cache_key)
        if cached is not None:
            return cached

    is_video = path.suffix.lower() in VIDEO_EXTENSIONS
    if not is_video:
        profile = MediaProfile(
            path=str(path),
            is_video=False,
            duration_seconds=0.0,
            total_frames=1,
            native_fps=0.0,
            width=0,
            height=0,
        )
    else:
        capture = cv2.VideoCapture(str(path))
        try:
            native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration_seconds = float(total_frames / native_fps) if native_fps > 0 and total_frames > 0 else 0.0
        finally:
            capture.release()
        profile = MediaProfile(
            path=str(path),
            is_video=True,
            duration_seconds=duration_seconds,
            total_frames=total_frames,
            native_fps=native_fps,
            width=width,
            height=height,
        )

    with _PROFILE_CACHE_LOCK:
        _PROFILE_CACHE[cache_key] = profile
    return profile


def adaptive_frame_plan(
    media_path: str | Path,
    *,
    purpose: str = "default",
    requested_sample_fps: float | None = None,
    requested_frames_per_video: int | None = None,
) -> tuple[float | None, int | None, MediaProfile]:
    profile = profile_media(media_path)
    if not profile.is_video:
        frames = 1 if requested_frames_per_video is None else max(1, min(int(requested_frames_per_video), 1))
        return requested_sample_fps, frames, profile

    duration = max(profile.duration_seconds, 0.0)
    if duration < 10.0:
        base_frames = 4
    elif duration < 60.0:
        base_frames = 3
    elif duration < 300.0:
        base_frames = 2
    else:
        base_frames = 1

    purpose_limit = {
        "detection": 4,
        "classification": 2,
        "embedding": 3,
        "reverse_search": 2,
        "layer3": 1,
        "preview": 1,
        "default": 3,
    }.get(str(purpose or "default").lower(), 3)
    effective_frames = max(1, min(base_frames, purpose_limit))
    if requested_frames_per_video is not None and requested_frames_per_video > 0:
        effective_frames = max(1, min(int(requested_frames_per_video), effective_frames))

    if duration > 0.0:
        target_fps = max(0.08, min(1.0, effective_frames / duration))
    else:
        target_fps = requested_sample_fps if requested_sample_fps and requested_sample_fps > 0 else 0.25

    if requested_sample_fps is not None and requested_sample_fps > 0:
        effective_sample_fps = min(float(requested_sample_fps), target_fps)
    else:
        effective_sample_fps = target_fps

    return effective_sample_fps, effective_frames, profile


def should_skip_reverse_search(
    media_path: str | Path,
    *,
    has_strong_internal_match: bool = False,
    system_under_load: bool = False,
) -> tuple[bool, str | None, MediaProfile]:
    profile = profile_media(media_path)
    if not profile.is_video:
        return has_strong_internal_match, (
            "Internal match already strong enough; public reverse search skipped."
            if has_strong_internal_match
            else None
        ), profile
    if has_strong_internal_match:
        return True, "Strong internal video match found; public reverse search skipped.", profile
    if system_under_load:
        return True, "System load is elevated; public reverse search skipped for this video.", profile
    if profile.duration_seconds > 300.0:
        return True, "Long video detected; public reverse search skipped to keep analysis responsive.", profile
    return False, None, profile
