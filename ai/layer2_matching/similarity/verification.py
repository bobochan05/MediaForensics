from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ai.layer2_matching.audio.audio_embedding import AudioEmbeddingService
from ai.layer2_matching.audio.audio_extract import extract_audio_from_media
from ai.layer1_detection.frame_extractor import extract_sampled_frames
from ai.layer2_matching.similarity.embedding import VisualEmbeddingService
from ai.shared.file_utils import ensure_dir


def cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.size == 0 or right.size == 0:
        return None
    if left.shape != right.shape:
        return None
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0 or right_norm <= 0:
        return None
    return float(np.dot(left / left_norm, right / right_norm))


def _phash_bits_from_media(media_path: str | Path) -> np.ndarray | None:
    try:
        frames = extract_sampled_frames(
            video_path=media_path,
            image_size=128,
            sample_fps=0.5,
            frames_per_video=1,
        )
    except Exception:
        return None

    if not frames:
        return None

    frame = frames[0].convert("L").resize((32, 32))
    pixels = np.asarray(frame, dtype=np.float32)
    dct = cv2.dct(pixels)
    low_freq = dct[:8, :8]
    flattened = low_freq.flatten()
    median = float(np.median(flattened[1:])) if flattened.size > 1 else float(np.median(flattened))
    return (low_freq > median).astype(np.uint8).reshape(-1)


def phash_difference(left_path: str | Path, right_path: str | Path) -> int | None:
    left_bits = _phash_bits_from_media(left_path)
    right_bits = _phash_bits_from_media(right_path)
    if left_bits is None or right_bits is None or left_bits.shape != right_bits.shape:
        return None
    return int(np.count_nonzero(left_bits != right_bits))


@dataclass
class VerificationResult:
    accepted: bool
    visual_similarity: float | None = None
    audio_similarity: float | None = None
    combined_score: float = 0.0
    visual_verified: bool = False
    audio_verified: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


class MultimodalVerificationService:
    def __init__(
        self,
        visual_embedder: VisualEmbeddingService,
        audio_embedder: AudioEmbeddingService,
        cache_dir: str | Path,
        visual_threshold: float = 0.8,
        audio_threshold: float = 0.75,
        visual_weight: float = 0.65,
        audio_weight: float = 0.35,
    ) -> None:
        self.visual_embedder = visual_embedder
        self.audio_embedder = audio_embedder
        self.cache_dir = ensure_dir(cache_dir)
        self.visual_threshold = float(visual_threshold)
        self.audio_threshold = float(audio_threshold)
        self.visual_weight = float(visual_weight)
        self.audio_weight = float(audio_weight)

    def verify_candidate(
        self,
        candidate_path: str | Path,
        original_visual_embedding: np.ndarray | None,
        original_audio_embedding: np.ndarray | None = None,
        original_media_path: str | Path | None = None,
    ) -> VerificationResult:
        candidate_path = Path(candidate_path)
        visual_similarity: float | None = None
        audio_similarity: float | None = None
        verification_metadata: dict[str, object] = {"candidate_path": str(candidate_path)}
        phash_diff: int | None = None

        if original_visual_embedding is not None:
            try:
                candidate_visual_embedding = self.visual_embedder.embed_media(candidate_path)
                visual_similarity = cosine_similarity(original_visual_embedding, candidate_visual_embedding)
            except Exception as exc:
                verification_metadata["visual_error"] = str(exc)

        if original_media_path is not None:
            try:
                phash_diff = phash_difference(original_media_path, candidate_path)
            except Exception:
                phash_diff = None

        if original_audio_embedding is not None:
            try:
                extracted_audio = extract_audio_from_media(candidate_path, self.cache_dir / "audio")
                if extracted_audio.has_audio and extracted_audio.waveform is not None and extracted_audio.sample_rate is not None:
                    candidate_audio = self.audio_embedder.embed_audio(
                        waveform=extracted_audio.waveform,
                        sample_rate=extracted_audio.sample_rate,
                        duration_seconds=extracted_audio.duration_seconds,
                    )
                    audio_similarity = cosine_similarity(original_audio_embedding, candidate_audio.combined_embedding)
            except Exception as exc:
                verification_metadata["audio_error"] = str(exc)

        visual_verified = bool(visual_similarity is not None and visual_similarity >= self.visual_threshold)
        audio_verified = bool(audio_similarity is not None and audio_similarity >= self.audio_threshold)

        weighted_parts: list[tuple[float, float]] = []
        if visual_similarity is not None:
            weighted_parts.append((self.visual_weight, visual_similarity))
        if audio_similarity is not None:
            weighted_parts.append((self.audio_weight, audio_similarity))
        if weighted_parts:
            total_weight = sum(weight for weight, _ in weighted_parts)
            combined_score = float(sum(weight * score for weight, score in weighted_parts) / max(total_weight, 1e-6))
        else:
            combined_score = 0.0

        accepted = visual_verified or audio_verified
        verification_metadata["visual_threshold"] = self.visual_threshold
        verification_metadata["audio_threshold"] = self.audio_threshold
        verification_metadata["phash_diff"] = phash_diff
        return VerificationResult(
            accepted=accepted,
            visual_similarity=visual_similarity,
            audio_similarity=audio_similarity,
            combined_score=combined_score,
            visual_verified=visual_verified,
            audio_verified=audio_verified,
            metadata=verification_metadata,
        )
