from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel as HFCLIPModel
from transformers import CLIPProcessor


class ClipModel(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = self._load_processor(model_name)
        self.model = self._load_model(model_name).to(self.device)
        self.feature_dim = int(self.model.visual_projection.out_features)
        self.model.eval()

    @staticmethod
    def _project_if_needed(tensor: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        if tensor.dim() > 2:
            tensor = tensor.mean(dim=1)
        tensor = tensor.float()
        in_features = int(getattr(projection, "in_features", tensor.shape[-1]))
        out_features = int(getattr(projection, "out_features", tensor.shape[-1]))
        feature_width = int(tensor.shape[-1])
        if feature_width == out_features:
            return tensor
        if feature_width == in_features:
            return projection(tensor)
        raise ValueError(
            f"Unexpected CLIP feature width {feature_width}; expected {in_features} or {out_features}."
        )

    @staticmethod
    def _load_processor(model_name: str) -> CLIPProcessor:
        try:
            return CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        except OSError:
            if os.getenv("DEEPFAKE_ALLOW_MODEL_DOWNLOADS") == "1":
                return CLIPProcessor.from_pretrained(model_name)
            raise

    @staticmethod
    def _load_model(model_name: str) -> HFCLIPModel:
        try:
            return HFCLIPModel.from_pretrained(model_name, local_files_only=True, use_safetensors=False)
        except OSError:
            if os.getenv("DEEPFAKE_ALLOW_MODEL_DOWNLOADS") == "1":
                return HFCLIPModel.from_pretrained(model_name, use_safetensors=False)
            raise

    def _coerce_image_features(self, output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return self._project_if_needed(output, self.model.visual_projection)
        if hasattr(output, "image_embeds") and isinstance(output.image_embeds, torch.Tensor):
            return self._project_if_needed(output.image_embeds, self.model.visual_projection)
        if hasattr(output, "pooler_output") and isinstance(output.pooler_output, torch.Tensor):
            return self._project_if_needed(output.pooler_output, self.model.visual_projection)
        if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
            return self._project_if_needed(output.last_hidden_state, self.model.visual_projection)
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return self._project_if_needed(output[0], self.model.visual_projection)
        raise TypeError(f"Unsupported CLIP image feature output type: {type(output)!r}")

    def _coerce_text_features(self, output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return self._project_if_needed(output, self.model.text_projection)
        if hasattr(output, "text_embeds") and isinstance(output.text_embeds, torch.Tensor):
            return self._project_if_needed(output.text_embeds, self.model.text_projection)
        if hasattr(output, "pooler_output") and isinstance(output.pooler_output, torch.Tensor):
            return self._project_if_needed(output.pooler_output, self.model.text_projection)
        if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
            return self._project_if_needed(output.last_hidden_state, self.model.text_projection)
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return self._project_if_needed(output[0], self.model.text_projection)
        raise TypeError(f"Unsupported CLIP text feature output type: {type(output)!r}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raw_output = self.model.get_image_features(pixel_values=pixel_values)
        image_features = self._coerce_image_features(raw_output).float()
        return F.normalize(image_features, p=2, dim=1)

    @torch.inference_mode()
    def extract_features(self, images: Sequence[Image.Image]) -> np.ndarray:
        rgb_images = [image.convert("RGB") for image in images]
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        features = self.forward(pixel_values)
        return features.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def extract_text_features(self, texts: Sequence[str]) -> np.ndarray:
        prompts = [str(text or "").strip() for text in texts]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        text_features = self._coerce_text_features(self.model.get_text_features(**inputs)).float()
        text_features = F.normalize(text_features, p=2, dim=1)
        return text_features.cpu().numpy().astype(np.float32)
