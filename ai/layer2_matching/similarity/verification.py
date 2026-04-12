from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ai.layer2_matching.audio.audio_embedding import AudioEmbeddingService
from ai.layer2_matching.audio.audio_extract import extract_audio_from_media
from ai.layer1_detection.frame_extractor import extract_sampled_frames
from ai.layer2_matching.similarity.embedding import VisualEmbeddingService
from ai.shared.file_utils import ensure_dir


LOGGER = logging.getLogger(__name__)
HASH_PREPROCESS_SIZE = 256
EMBED_PREPROCESS_SIZE = 224
PATCH_GRID_SIZE = 4
EXACT_HASH_THRESHOLD = 5
NEAR_EXACT_HASH_THRESHOLD = 10
PARTIAL_PATCH_RATIO_THRESHOLD = 0.35
HIGH_EMBEDDING_THRESHOLD = 0.75
MEDIUM_EMBEDDING_THRESHOLD = 0.5
_HASH_CACHE: dict[tuple[str, int, int], dict[str, object]] = {}
_EMBEDDING_CACHE: dict[tuple[str, int, int], np.ndarray] = {}


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


def _normalized_embedding_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    raw_score = cosine_similarity(left, right)
    if raw_score is None:
        return None
    return float(max(0.0, min(1.0, (raw_score + 1.0) / 2.0)))


def _media_signature(media_path: str | Path) -> tuple[str, int]:
    path = Path(media_path).resolve()
    stat = path.stat()
    return str(path), int(stat.st_mtime_ns)


def _extract_reference_frame(media_path: str | Path, image_size: int = HASH_PREPROCESS_SIZE) -> Image.Image | None:
    try:
        frames = extract_sampled_frames(
            video_path=media_path,
            image_size=image_size,
            sample_fps=0.5,
            frames_per_video=1,
        )
    except Exception:
        return None

    if not frames:
        return None

    return frames[0]


def _preprocess_grayscale(image: Image.Image, size: int = HASH_PREPROCESS_SIZE) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    normalized = ImageOps.autocontrast(grayscale)
    normalized = ImageOps.equalize(normalized)
    return normalized.resize((size, size), Image.Resampling.LANCZOS)


def _preprocess_rgb(image: Image.Image, size: int = EMBED_PREPROCESS_SIZE) -> Image.Image:
    rgb = image.convert("RGB")
    normalized = ImageOps.autocontrast(rgb)
    return normalized.resize((size, size), Image.Resampling.LANCZOS)


def _phash_bits_from_image(image: Image.Image) -> np.ndarray:
    frame = _preprocess_grayscale(image, size=32)
    pixels = np.asarray(frame, dtype=np.float32)
    dct = cv2.dct(pixels)
    low_freq = dct[:8, :8]
    flattened = low_freq.flatten()
    median = float(np.median(flattened[1:])) if flattened.size > 1 else float(np.median(flattened))
    return (low_freq > median).astype(np.uint8).reshape(-1)


def _phash_bits_from_media(media_path: str | Path) -> np.ndarray | None:
    frame = _extract_reference_frame(media_path)
    if frame is None:
        return None
    return _phash_bits_from_image(frame)


def _ahash_bits_from_image(image: Image.Image) -> np.ndarray:
    grayscale = _preprocess_grayscale(image, size=8)
    pixels = np.asarray(grayscale, dtype=np.float32)
    mean_value = float(np.mean(pixels))
    return (pixels >= mean_value).astype(np.uint8).reshape(-1)


def _dhash_bits_from_image(image: Image.Image) -> np.ndarray:
    grayscale = _preprocess_grayscale(image, size=9)
    pixels = np.asarray(grayscale, dtype=np.float32)
    diff = pixels[:, 1:] >= pixels[:, :-1]
    return diff.astype(np.uint8).reshape(-1)


def _hamming_distance(left_bits: np.ndarray, right_bits: np.ndarray) -> int | None:
    if left_bits.shape != right_bits.shape:
        return None
    return int(np.count_nonzero(left_bits != right_bits))


def _bits_to_hex(bits: np.ndarray) -> str:
    if bits.size == 0:
        return ""
    packed = np.packbits(bits.astype(np.uint8))
    return bytes(packed.tolist()).hex()


def _patch_hashes(image: Image.Image, grid_size: int = PATCH_GRID_SIZE) -> list[np.ndarray]:
    processed = _preprocess_grayscale(image, size=HASH_PREPROCESS_SIZE)
    width, height = processed.size
    patch_width = max(1, width // grid_size)
    patch_height = max(1, height // grid_size)
    hashes: list[np.ndarray] = []
    for row in range(grid_size):
        for col in range(grid_size):
            left = col * patch_width
            top = row * patch_height
            right = width if col == grid_size - 1 else (col + 1) * patch_width
            bottom = height if row == grid_size - 1 else (row + 1) * patch_height
            patch = processed.crop((left, top, right, bottom)).resize((32, 32), Image.Resampling.LANCZOS)
            hashes.append(_phash_bits_from_image(patch))
    return hashes


def _patch_match_ratio(left_hashes: list[np.ndarray], right_hashes: list[np.ndarray], max_distance: int = NEAR_EXACT_HASH_THRESHOLD) -> float:
    if not left_hashes or not right_hashes:
        return 0.0
    matched = 0
    remaining = list(right_hashes)
    for left_hash in left_hashes:
        best_index = None
        best_distance = None
        for index, right_hash in enumerate(remaining):
            distance = _hamming_distance(left_hash, right_hash)
            if distance is None:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = index
        if best_distance is not None and best_distance <= max_distance and best_index is not None:
            matched += 1
            remaining.pop(best_index)
    return float(matched / max(len(left_hashes), 1))


def _hash_bundle_for_media(media_path: str | Path) -> dict[str, object] | None:
    key = (*_media_signature(media_path), HASH_PREPROCESS_SIZE)
    cached = _HASH_CACHE.get(key)
    if cached is not None:
        return cached

    frame = _extract_reference_frame(media_path, image_size=HASH_PREPROCESS_SIZE)
    if frame is None:
        return None

    processed = _preprocess_grayscale(frame)
    pixels = np.asarray(processed, dtype=np.uint8)
    phash_bits = _phash_bits_from_image(processed)
    dhash_bits = _dhash_bits_from_image(processed)
    ahash_bits = _ahash_bits_from_image(processed)
    patch_hashes = _patch_hashes(processed)
    bundle = {
        "phash_bits": phash_bits,
        "dhash_bits": dhash_bits,
        "ahash_bits": ahash_bits,
        "patch_hashes": patch_hashes,
        "phash_hex": _bits_to_hex(phash_bits),
        "dhash_hex": _bits_to_hex(dhash_bits),
        "ahash_hex": _bits_to_hex(ahash_bits),
        "md5": hashlib.md5(pixels.tobytes()).hexdigest(),
    }
    _HASH_CACHE[key] = bundle
    return bundle


def _preprocessed_embedding_for_media(media_path: str | Path, visual_embedder: VisualEmbeddingService) -> np.ndarray | None:
    key = (*_media_signature(media_path), EMBED_PREPROCESS_SIZE)
    cached = _EMBEDDING_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        frames = extract_sampled_frames(
            video_path=media_path,
            image_size=EMBED_PREPROCESS_SIZE,
            sample_fps=visual_embedder.sample_fps,
            frames_per_video=visual_embedder.max_frames_per_video,
        )
    except Exception:
        return None
    if not frames:
        return None
    processed_frames = [_preprocess_rgb(frame, size=visual_embedder.image_size) for frame in frames]
    embedding = visual_embedder.embed_images(processed_frames)
    _EMBEDDING_CACHE[key] = embedding
    return embedding


def phash_difference(left_path: str | Path, right_path: str | Path) -> int | None:
    left_bundle = _hash_bundle_for_media(left_path)
    right_bundle = _hash_bundle_for_media(right_path)
    if left_bundle is None or right_bundle is None:
        return None
    return _hamming_distance(left_bundle["phash_bits"], right_bundle["phash_bits"])


@dataclass
class VerificationResult:
    accepted: bool
    visual_similarity: float | None = None
    audio_similarity: float | None = None
    combined_score: float = 0.0
    match_type: str = "related"
    confidence_label: str = "LOW"
    hash_distance: int | None = None
    embedding_score: float | None = None
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
        dhash_diff: int | None = None
        ahash_diff: int | None = None
        patch_match_ratio = 0.0
        match_type = "related"
        confidence_label = "LOW"
        explanation = "Contextually related result surfaced through embedding similarity."
        hash_score = 0.0

        original_preprocessed_embedding = None
        if original_media_path is not None:
            try:
                original_preprocessed_embedding = _preprocessed_embedding_for_media(original_media_path, self.visual_embedder)
            except Exception as exc:
                verification_metadata["original_visual_error"] = str(exc)
        if original_preprocessed_embedding is None:
            original_preprocessed_embedding = original_visual_embedding

        with ThreadPoolExecutor(max_workers=2) as executor:
            visual_future = None
            hash_future = None
            if original_preprocessed_embedding is not None:
                visual_future = executor.submit(_preprocessed_embedding_for_media, candidate_path, self.visual_embedder)
            if original_media_path is not None:
                hash_future = executor.submit(_hash_bundle_for_media, candidate_path)
                try:
                    original_hash_bundle = _hash_bundle_for_media(original_media_path)
                except Exception as exc:
                    original_hash_bundle = None
                    verification_metadata["hash_error"] = str(exc)
            else:
                original_hash_bundle = None

            if visual_future is not None:
                try:
                    candidate_visual_embedding = visual_future.result()
                    visual_similarity = _normalized_embedding_similarity(original_preprocessed_embedding, candidate_visual_embedding)
                except Exception as exc:
                    verification_metadata["visual_error"] = str(exc)

            candidate_hash_bundle = None
            if hash_future is not None:
                try:
                    candidate_hash_bundle = hash_future.result()
                except Exception as exc:
                    verification_metadata["hash_error"] = str(exc)

        if original_hash_bundle is not None and candidate_hash_bundle is not None:
            phash_diff = _hamming_distance(original_hash_bundle["phash_bits"], candidate_hash_bundle["phash_bits"])
            dhash_diff = _hamming_distance(original_hash_bundle["dhash_bits"], candidate_hash_bundle["dhash_bits"])
            ahash_diff = _hamming_distance(original_hash_bundle["ahash_bits"], candidate_hash_bundle["ahash_bits"])
            patch_match_ratio = _patch_match_ratio(
                list(original_hash_bundle["patch_hashes"]),
                list(candidate_hash_bundle["patch_hashes"]),
            )
            verification_metadata["phash"] = original_hash_bundle["phash_hex"]
            verification_metadata["candidate_phash"] = candidate_hash_bundle["phash_hex"]
            verification_metadata["dhash"] = original_hash_bundle["dhash_hex"]
            verification_metadata["candidate_dhash"] = candidate_hash_bundle["dhash_hex"]
            verification_metadata["ahash"] = original_hash_bundle["ahash_hex"]
            verification_metadata["candidate_ahash"] = candidate_hash_bundle["ahash_hex"]
            verification_metadata["patch_match_ratio"] = round(patch_match_ratio, 4)
            verification_metadata["candidate_md5"] = candidate_hash_bundle["md5"]
            verification_metadata["query_md5"] = original_hash_bundle["md5"]

            if original_hash_bundle["md5"] == candidate_hash_bundle["md5"] or (phash_diff is not None and phash_diff <= EXACT_HASH_THRESHOLD):
                match_type = "exact"
                confidence_label = "HIGH"
                hash_score = 1.0
                explanation = "Exact or near-identical image detected using perceptual hashing."
            elif phash_diff is not None and phash_diff <= NEAR_EXACT_HASH_THRESHOLD:
                match_type = "near_exact"
                confidence_label = "HIGH"
                hash_score = 0.9
                explanation = "Near-identical image with minor changes such as resize, compression, or watermark."
            elif patch_match_ratio >= PARTIAL_PATCH_RATIO_THRESHOLD:
                match_type = "visual"
                confidence_label = "MEDIUM"
                hash_score = min(0.85, 0.55 + patch_match_ratio * 0.6)
                explanation = "Detected despite crop, compression, or resizing differences through patch-based hashing."

        if original_audio_embedding is not None:
            try:
                extracted_audio = extract_audio_from_media(candidate_path, self.cache_dir / "audio")
                if extracted_audio.has_audio and extracted_audio.waveform is not None and extracted_audio.sample_rate is not None:
                    candidate_audio = self.audio_embedder.embed_audio(
                        waveform=extracted_audio.waveform,
                        sample_rate=extracted_audio.sample_rate,
                        duration_seconds=extracted_audio.duration_seconds,
                    )
                    audio_similarity = _normalized_embedding_similarity(original_audio_embedding, candidate_audio.combined_embedding)
            except Exception as exc:
                verification_metadata["audio_error"] = str(exc)

        embedding_score = float(visual_similarity) if visual_similarity is not None else 0.0
        if match_type not in {"exact", "near_exact"}:
            if embedding_score >= HIGH_EMBEDDING_THRESHOLD:
                match_type = "visual"
                confidence_label = "HIGH"
                explanation = "Strong visual match detected through normalized embedding similarity."
            elif embedding_score >= MEDIUM_EMBEDDING_THRESHOLD:
                match_type = "related"
                confidence_label = "MEDIUM"
                explanation = "Related image detected through normalized embedding similarity."
            elif patch_match_ratio >= PARTIAL_PATCH_RATIO_THRESHOLD:
                match_type = "visual"
                confidence_label = "MEDIUM"
            else:
                match_type = "related"
                confidence_label = "LOW"
                explanation = "Weak similarity signal only; no reliable exact or near-exact evidence was found."

        visual_verified = bool(
            match_type in {"exact", "near_exact", "visual"}
            or (visual_similarity is not None and visual_similarity >= MEDIUM_EMBEDDING_THRESHOLD)
        )
        audio_verified = bool(audio_similarity is not None and audio_similarity >= self.audio_threshold)
        combined_score = float((hash_score * 0.6) + (embedding_score * 0.4))
        if match_type == "exact":
            combined_score = max(combined_score, 0.98)
        elif match_type == "near_exact":
            combined_score = max(combined_score, 0.9)
        elif match_type == "visual":
            combined_score = max(combined_score, embedding_score)

        accepted = match_type in {"exact", "near_exact", "visual", "related"} and (combined_score >= MEDIUM_EMBEDDING_THRESHOLD or hash_score > 0.0 or audio_verified)
        verification_metadata["visual_threshold"] = self.visual_threshold
        verification_metadata["audio_threshold"] = self.audio_threshold
        verification_metadata["phash_diff"] = phash_diff
        verification_metadata["dhash_diff"] = dhash_diff
        verification_metadata["ahash_diff"] = ahash_diff
        verification_metadata["hash_distance"] = phash_diff
        verification_metadata["embedding_similarity"] = embedding_score
        verification_metadata["embedding_score"] = embedding_score
        verification_metadata["hash_score"] = hash_score
        verification_metadata["final_score"] = combined_score
        verification_metadata["match_type"] = match_type
        verification_metadata["confidence_label"] = confidence_label
        verification_metadata["explanation"] = explanation
        verification_metadata["partial_match"] = bool(patch_match_ratio >= PARTIAL_PATCH_RATIO_THRESHOLD)
        LOGGER.debug(
            "layer2_verification candidate=%s match_type=%s phash=%s dhash=%s ahash=%s embed=%.4f final=%.4f partial=%.4f",
            str(candidate_path),
            match_type,
            phash_diff,
            dhash_diff,
            ahash_diff,
            embedding_score,
            combined_score,
            patch_match_ratio,
        )
        return VerificationResult(
            accepted=accepted,
            visual_similarity=visual_similarity,
            audio_similarity=audio_similarity,
            combined_score=combined_score,
            match_type=match_type,
            confidence_label=confidence_label,
            hash_distance=phash_diff,
            embedding_score=embedding_score,
            visual_verified=visual_verified,
            audio_verified=audio_verified,
            metadata=verification_metadata,
        )
