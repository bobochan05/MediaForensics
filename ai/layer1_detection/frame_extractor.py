from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import List

import cv2
import numpy as np
from PIL import Image

from ai.shared.video_budget import adaptive_frame_plan

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
FRAME_DEDUP_HASH_THRESHOLD = 6
SCENE_CHANGE_THRESHOLD = 22.0
MIN_FRAME_SHARPNESS = 18.0
_FRAME_CACHE: dict[tuple[str, int, float | None, int | None, str], list[Image.Image]] = {}
_FRAME_CACHE_LOCK = Lock()


def extract_sampled_frames(
    video_path: str | Path,
    image_size: int = 224,
    sample_fps: float | None = 0.5,
    frames_per_video: int | None = None,
    purpose: str = "default",
) -> List[Image.Image]:
    video_path = Path(video_path)
    if video_path.suffix.lower() in IMAGE_EXTENSIONS:
        image = Image.open(video_path).convert("RGB")
        image = image.resize((image_size, image_size), Image.LANCZOS)
        if frames_per_video is not None and frames_per_video > 0:
            return [image.copy() for _ in range(frames_per_video)]
        return [image]

    effective_sample_fps, effective_frames_per_video, profile = adaptive_frame_plan(
        video_path,
        purpose=purpose,
        requested_sample_fps=sample_fps,
        requested_frames_per_video=frames_per_video,
    )
    try:
        cache_signature = int(video_path.stat().st_mtime_ns)
    except FileNotFoundError:
        cache_signature = 0
    cache_key = (
        str(video_path.resolve()),
        cache_signature,
        int(image_size),
        round(float(effective_sample_fps or 0.0), 4) if effective_sample_fps is not None else None,
        int(effective_frames_per_video) if effective_frames_per_video is not None else None,
        str(purpose or "default"),
    )
    with _FRAME_CACHE_LOCK:
        cached = _FRAME_CACHE.get(cache_key)
        if cached is not None:
            return [frame.copy() for frame in cached]

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        native_fps = float(profile.native_fps or capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(profile.total_frames or capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_indices = _compute_frame_indices(total_frames, native_fps, effective_sample_fps, effective_frames_per_video)

        frames: List[Image.Image] = []
        accepted_hashes: list[np.ndarray] = []
        accepted_signatures: list[np.ndarray] = []
        fallback_frame: Image.Image | None = None
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            candidate_image = Image.fromarray(frame)
            if fallback_frame is None:
                fallback_frame = candidate_image.copy()

            candidate_hash = _frame_phash_bits(frame)
            candidate_signature = _frame_signature(frame)
            if not _is_quality_frame(frame) and fallback_frame is not None and frames:
                continue
            if _is_near_duplicate(
                candidate_hash,
                accepted_hashes,
                candidate_signature,
                accepted_signatures,
            ):
                continue

            frames.append(candidate_image)
            accepted_hashes.append(candidate_hash)
            accepted_signatures.append(candidate_signature)

        result = frames if frames else ([fallback_frame] if fallback_frame is not None else [])
        with _FRAME_CACHE_LOCK:
            _FRAME_CACHE[cache_key] = [frame.copy() for frame in result]
        return result
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


def _frame_signature(frame: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(grayscale, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)


def _is_quality_frame(frame: np.ndarray) -> bool:
    grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
    if sharpness >= MIN_FRAME_SHARPNESS:
        return True
    brightness = float(np.mean(grayscale))
    return brightness >= 20.0


def _frame_phash_bits(frame: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    reduced = cv2.resize(grayscale, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(reduced)
    low_freq = dct[:8, :8]
    flattened = low_freq.flatten()
    median = float(np.median(flattened[1:])) if flattened.size > 1 else float(np.median(flattened))
    return (low_freq > median).astype(np.uint8).reshape(-1)


def _hamming_distance(left_bits: np.ndarray, right_bits: np.ndarray) -> int:
    return int(np.count_nonzero(left_bits != right_bits))


def _scene_delta(left_signature: np.ndarray, right_signature: np.ndarray) -> float:
    return float(np.mean(np.abs(left_signature - right_signature)))


def _is_near_duplicate(
    candidate_hash: np.ndarray,
    accepted_hashes: list[np.ndarray],
    candidate_signature: np.ndarray,
    accepted_signatures: list[np.ndarray],
) -> bool:
    for accepted_hash, accepted_signature in zip(accepted_hashes, accepted_signatures, strict=False):
        if _hamming_distance(candidate_hash, accepted_hash) <= FRAME_DEDUP_HASH_THRESHOLD:
            return True
        if _scene_delta(candidate_signature, accepted_signature) < SCENE_CHANGE_THRESHOLD:
            return True
    return False
