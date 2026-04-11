from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png"}
REAL_LABEL = 0
FAKE_LABEL = 1


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    video_path: Path
    label: int
    source: str


def collect_faceforensics_videos(
    dataset_dir: str | Path,
    max_real_videos: int | None = None,
    max_fake_videos: int | None = None,
    random_state: int = 42,
) -> List[VideoRecord]:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Invalid dataset_dir: {dataset_path}")

    # Support multiple common dataset layouts:
    # 1) FaceForensics++ layout: original_sequences/ and manipulated_sequences/
    # 2) Simple layout: real/ and fake/
    original_ffpp = dataset_path / "original_sequences"
    manipulated_ffpp = dataset_path / "manipulated_sequences"
    real_dir = dataset_path / "real"
    fake_dir = dataset_path / "fake"

    if original_ffpp.exists() and manipulated_ffpp.exists():
        original_root = original_ffpp
        manipulated_root = manipulated_ffpp
    elif real_dir.exists() and fake_dir.exists():
        original_root = real_dir
        manipulated_root = fake_dir
    else:
        raise FileNotFoundError(
            f"Dataset folder must contain either 'original_sequences' and 'manipulated_sequences',\n"
            f"or 'real' and 'fake'. Checked: {original_ffpp}, {manipulated_ffpp}, {real_dir}, {fake_dir}"
        )

    real_candidates = _find_video_files(original_root)
    fake_candidates = _find_video_files(manipulated_root)

    print(f"[DEBUG] original_sequences exists: {original_root.exists()} -> {original_root}")
    print(f"[DEBUG] manipulated_sequences exists: {manipulated_root.exists()} -> {manipulated_root}")
    print(f"[DEBUG] Found {len(real_candidates)} real videos before sampling")
    print(f"[DEBUG] Found {len(fake_candidates)} fake videos before sampling")

    if not real_candidates:
        raise ValueError(f"No videos found under: {original_root}")
    if not fake_candidates:
        raise ValueError(f"No videos found under: {manipulated_root}")

    real_paths = _sample_video_paths(real_candidates, max_real_videos, random_state)
    fake_paths = _sample_video_paths(fake_candidates, max_fake_videos, random_state + 1)

    print(f"[DEBUG] Using {len(real_paths)} real videos after sampling")
    print(f"[DEBUG] Using {len(fake_paths)} fake videos after sampling")

    records: List[VideoRecord] = []

    for video_path in real_paths:
        video_id = video_path.relative_to(dataset_path).as_posix()
        records.append(
            VideoRecord(
                video_id=video_id,
                video_path=video_path,
                label=REAL_LABEL,
                source="original",
            )
        )

    for video_path in fake_paths:
        relative_path = video_path.relative_to(dataset_path)
        manipulation_name = relative_path.parts[1] if len(relative_path.parts) > 1 else "manipulated"
        records.append(
            VideoRecord(
                video_id=relative_path.as_posix(),
                video_path=video_path,
                label=FAKE_LABEL,
                source=manipulation_name,
            )
        )

    if not records:
        raise ValueError(f"No videos found under {dataset_path}. Check FaceForensics++ folder structure and video extensions.")

    return sorted(records, key=lambda record: record.video_id)


def summarize_records(video_records: Iterable[VideoRecord]) -> dict[str, int]:
    records = list(video_records)
    return {
        "total_videos": len(records),
        "real_videos": sum(record.label == REAL_LABEL for record in records),
        "fake_videos": sum(record.label == FAKE_LABEL for record in records),
    }


def _find_video_files(root_dir: Path) -> List[Path]:
    video_paths = [
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(video_paths)


def _sample_video_paths(video_paths: List[Path], max_videos: int | None, random_state: int) -> List[Path]:
    if max_videos is None or max_videos >= len(video_paths):
        return video_paths

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(video_paths), size=max_videos, replace=False)
    return [video_paths[index] for index in sorted(indices)]
