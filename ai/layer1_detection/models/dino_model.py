from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOModel(nn.Module):
    def __init__(self, model_path: str | Path | None = None, device: str | None = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.feature_dim = int(getattr(self.backbone, "embed_dim", 384))

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

    def forward(self, tensors: torch.Tensor) -> torch.Tensor:
        embeddings = self.backbone(tensors)
        return F.normalize(embeddings, p=2, dim=1)

    def extract_features_from_tensors(self, tensors: torch.Tensor) -> torch.Tensor:
        return self.forward(tensors)

    def freeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze_last_layers(self) -> None:
        self.freeze_all()
        if hasattr(self.backbone, "blocks") and len(self.backbone.blocks) > 0:
            for parameter in self.backbone.blocks[-1].parameters():
                parameter.requires_grad = True
        if hasattr(self.backbone, "norm"):
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad = True

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def train_mode(self) -> None:
        self.backbone.eval()
        if hasattr(self.backbone, "blocks") and len(self.backbone.blocks) > 0:
            if any(parameter.requires_grad for parameter in self.backbone.blocks[-1].parameters()):
                self.backbone.blocks[-1].train()
        if hasattr(self.backbone, "norm") and any(parameter.requires_grad for parameter in self.backbone.norm.parameters()):
            self.backbone.norm.train()

    def eval_mode(self) -> None:
        self.backbone.eval()

    def save(self, path: str | Path) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_version": 2,
                "state_dict": {key: value.detach().cpu() for key, value in self.backbone.state_dict().items()},
                "feature_dim": self.feature_dim,
            },
            path_obj,
        )

    def load(self, path: str | Path) -> None:
        path_obj = Path(path)
        payload = torch.load(path_obj, map_location=self.device)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        self.backbone.load_state_dict(state_dict)

    @torch.inference_mode()
    def extract_features(self, images: Sequence[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        features = self.extract_features_from_tensors(tensors)
        return features.cpu().numpy().astype(np.float32)
