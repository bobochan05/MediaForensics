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

    @staticmethod
    def _unwrap_feature_tensor(output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
            value = getattr(output, attr, None)
            if isinstance(value, torch.Tensor):
                return value
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        raise TypeError(f"Unsupported CLIP feature output type: {type(output)!r}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raw_features = self.model.get_image_features(pixel_values=pixel_values)
        image_features = self._unwrap_feature_tensor(raw_features).float()
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
        raw_features = self.model.get_text_features(**inputs)
        text_features = self._unwrap_feature_tensor(raw_features).float()
        text_features = F.normalize(text_features, p=2, dim=1)
        return text_features.cpu().numpy().astype(np.float32)
