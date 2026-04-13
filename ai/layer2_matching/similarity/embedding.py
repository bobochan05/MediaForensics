from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from ai.layer1_detection.frame_extractor import extract_sampled_frames
from ai.layer1_detection.models.clip_model import ClipModel
from ai.shared.video_budget import adaptive_frame_plan


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


class VisualEmbeddingService:
    def __init__(
        self,
        device: str = "auto",
        sample_fps: float = 0.5,
        max_frames_per_video: int = 8,
        image_size: int = 224,
    ) -> None:
        self.requested_device = device
        self.sample_fps = sample_fps
        self.max_frames_per_video = max_frames_per_video
        self.image_size = image_size
        self._model: ClipModel | None = None
        self._device: str | None = None

    def _resolve_device(self) -> str:
        if self.requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.requested_device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.requested_device

    def _model_instance(self) -> ClipModel:
        if self._model is None:
            self._device = self._resolve_device()
            self._model = ClipModel(device=self._device)
        return self._model

    @staticmethod
    def _center_crop(image: Image.Image, ratio: float) -> Image.Image:
        width, height = image.size
        crop_width = max(1, int(width * ratio))
        crop_height = max(1, int(height * ratio))
        left = max((width - crop_width) // 2, 0)
        top = max((height - crop_height) // 2, 0)
        return image.crop((left, top, left + crop_width, top + crop_height)).resize((width, height), Image.LANCZOS)

    def _generate_views(self, image: Image.Image) -> list[Image.Image]:
        base = image.convert("RGB")
        return [base]

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("At least one image is required to compute a visual embedding.")

        model = self._model_instance()
        grouped_views = [self._generate_views(image) for image in images]
        flattened_views = [view for group in grouped_views for view in group]
        features = model.extract_features(flattened_views)

        frame_embeddings: list[np.ndarray] = []
        offset = 0
        for group in grouped_views:
            group_features = features[offset : offset + len(group)]
            offset += len(group)
            frame_embeddings.append(_normalize(group_features.mean(axis=0)))

        return _normalize(np.mean(frame_embeddings, axis=0))

    def embed_media(self, media_path: str | Path) -> np.ndarray:
        effective_sample_fps, effective_frames_per_video, _ = adaptive_frame_plan(
            media_path,
            purpose="embedding",
            requested_sample_fps=self.sample_fps,
            requested_frames_per_video=self.max_frames_per_video,
        )
        frames = extract_sampled_frames(
            video_path=media_path,
            image_size=self.image_size,
            sample_fps=effective_sample_fps,
            frames_per_video=effective_frames_per_video,
            purpose="embedding",
        )
        if not frames:
            raise ValueError(f"No frames available for visual embedding: {media_path}")
        return self.embed_images(frames)

