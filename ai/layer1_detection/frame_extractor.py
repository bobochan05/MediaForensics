from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def extract_sampled_frames(
    video_path: str | Path,
    image_size: int = 224,
    sample_fps: float | None = 0.5,
    frames_per_video: int | None = None,
) -> List[Image.Image]:
    video_path = Path(video_path)
    if video_path.suffix.lower() in IMAGE_EXTENSIONS:
        image = Image.open(video_path).convert("RGB")
        image = image.resize((image_size, image_size), Image.LANCZOS)
        if frames_per_video is not None and frames_per_video > 0:
            return [image.copy() for _ in range(frames_per_video)]
        return [image]

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_indices = _compute_frame_indices(total_frames, native_fps, sample_fps, frames_per_video)

        frames: List[Image.Image] = []
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(frame))

        return frames
    finally:
        capture.release()


def _compute_frame_indices(
    total_frames: int,
    native_fps: float,
    sample_fps: float | None,
    frames_per_video: int | None,
) -> List[int]:
    if total_frames <= 0:
        return [0]

    if frames_per_video is not None and frames_per_video > 0:
        sample_count = min(frames_per_video, total_frames)
        return sorted(set(np.linspace(0, total_frames - 1, num=sample_count, dtype=int).tolist()))

    if sample_fps is not None and sample_fps > 0 and native_fps > 0:
        frame_step = max(int(round(native_fps / sample_fps)), 1)
        return list(range(0, total_frames, frame_step))

    fallback_count = min(10, total_frames)
    return sorted(set(np.linspace(0, total_frames - 1, num=fallback_count, dtype=int).tolist()))
