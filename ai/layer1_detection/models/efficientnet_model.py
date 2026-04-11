from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class EfficientNetModel(nn.Module):
    def __init__(self, model_path: str | Path | None = None, device: str | None = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)
        self.feature_dim = int(self.backbone.classifier[1].in_features)

        self.train_transform = T.Compose(
            [
                T.Resize((256, 256)),
                T.RandomResizedCrop((224, 224), scale=(0.85, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.eval_transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.transform = self.eval_transform

        self.to(self.device)

        if model_path:
            self.load(model_path)

        self.eval_mode()

    def _extract_raw_features(self, tensors: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(tensors)
        pooled = self.backbone.avgpool(features)
        return torch.flatten(pooled, 1)

    def forward(self, tensors: torch.Tensor) -> torch.Tensor:
        raw_features = self._extract_raw_features(tensors)
        return F.normalize(raw_features, p=2, dim=1)

    def extract_features_from_tensors(self, tensors: torch.Tensor) -> torch.Tensor:
        return self.forward(tensors)

    def freeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze_last_layers(self) -> None:
        self.freeze_all()
        for block in self.backbone.features[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def train_mode(self) -> None:
        self.backbone.eval()
        for block in self.backbone.features[-2:]:
            if any(parameter.requires_grad for parameter in block.parameters()):
                block.train()

    def eval_mode(self) -> None:
        self.backbone.eval()

    def save(self, path: str | Path) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.backbone.state_dict().items()
            if not key.startswith("classifier.")
        }
        torch.save(
            {
                "model_version": 2,
                "state_dict": state_dict,
                "feature_dim": self.feature_dim,
            },
            path_obj,
        )

    @staticmethod
    def _unwrap_state_dict(payload) -> dict[str, torch.Tensor]:
        if isinstance(payload, dict):
            for key in ("state_dict", "model_state_dict", "backbone_state_dict"):
                nested = payload.get(key)
                if isinstance(nested, dict):
                    return nested
        if isinstance(payload, dict):
            return payload
        raise TypeError(f"Unsupported EfficientNet checkpoint format: {type(payload)!r}")

    @staticmethod
    def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        normalized_state: dict[str, torch.Tensor] = {}
        for original_key, value in state_dict.items():
            key = original_key
            for prefix in ("module.", "backbone."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]

            if key.startswith("classifier.") or ".classifier." in f".{key}":
                continue

            normalized_state[key] = value
        return normalized_state

    def load(self, path: str | Path) -> None:
        path_obj = Path(path)
        payload = torch.load(path_obj, map_location=self.device)
        state_dict = self._unwrap_state_dict(payload)
        filtered_state_dict = self._normalize_state_dict_keys(state_dict)
        missing_keys, unexpected_keys = self.backbone.load_state_dict(filtered_state_dict, strict=False)
        missing_keys = [key for key in missing_keys if not key.startswith("classifier.")]
        unexpected_keys = [key for key in unexpected_keys if not key.startswith("classifier.")]
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "EfficientNet checkpoint is incompatible with the current feature extractor. "
                f"Missing keys: {missing_keys} | Unexpected keys: {unexpected_keys}"
            )

    @torch.inference_mode()
    def extract_features(self, images: Sequence[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        features = self.extract_features_from_tensors(tensors)
        return features.cpu().numpy().astype(np.float32)
