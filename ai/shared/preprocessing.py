from __future__ import annotations

from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def media_type_from_path(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return "unknown"


def ensure_within_project(path: str | Path, project_root: str | Path) -> Path:
    target = Path(path).resolve()
    root = Path(project_root).resolve()
    target.relative_to(root)
    return target

