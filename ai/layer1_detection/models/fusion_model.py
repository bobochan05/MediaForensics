from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ai.layer1_detection.models.fft_model import FFTBranch

MODEL_VERSION = 2


class FusionMLP(nn.Module):
    def __init__(
        self,
        clip_dim: int,
        dino_dim: int,
        efficientnet_dim: int,
        fft_dim: int = 128,
        hidden_dims: tuple[int, int] = (512, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.clip_dim = int(clip_dim)
        self.dino_dim = int(dino_dim)
        self.efficientnet_dim = int(efficientnet_dim)
        self.fft_dim = int(fft_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.dropout = float(dropout)
        self._printed_shapes = False
        self.branch_feature_dim = self.clip_dim + self.dino_dim + self.efficientnet_dim
        self.fusion_input_dim = self.branch_feature_dim + self.fft_dim

        self.fft_branch = FFTBranch(feature_dim=self.fft_dim)
        hidden_1, hidden_2 = self.hidden_dims
        self.net = nn.Sequential(
            nn.Linear(self.fusion_input_dim, hidden_1),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_2, 1),
        )

    def forward(
        self,
        clip_features: torch.Tensor,
        dino_features: torch.Tensor,
        efficientnet_features: torch.Tensor,
        frequency_inputs: torch.Tensor,
    ) -> torch.Tensor:
        if clip_features.ndim != 2 or clip_features.shape[1] != self.clip_dim:
            raise ValueError(f"Expected CLIP features with shape [N, {self.clip_dim}], got {tuple(clip_features.shape)}")
        if dino_features.ndim != 2 or dino_features.shape[1] != self.dino_dim:
            raise ValueError(f"Expected DINO features with shape [N, {self.dino_dim}], got {tuple(dino_features.shape)}")
        if efficientnet_features.ndim != 2 or efficientnet_features.shape[1] != self.efficientnet_dim:
            raise ValueError(
                f"Expected EfficientNet features with shape [N, {self.efficientnet_dim}], "
                f"got {tuple(efficientnet_features.shape)}"
            )
        if frequency_inputs.ndim == 3:
            frequency_inputs = frequency_inputs.unsqueeze(1)
        if frequency_inputs.ndim != 4 or frequency_inputs.shape[1] != 1:
            raise ValueError(
                "Expected FFT inputs with shape [N, 1, H, W], "
                f"got {tuple(frequency_inputs.shape)}"
            )

        batch_size = clip_features.shape[0]
        if dino_features.shape[0] != batch_size or efficientnet_features.shape[0] != batch_size or frequency_inputs.shape[0] != batch_size:
            raise ValueError(
                "All feature batches must have the same batch size, got "
                f"clip={clip_features.shape[0]}, dino={dino_features.shape[0]}, "
                f"efficientnet={efficientnet_features.shape[0]}, fft={frequency_inputs.shape[0]}"
            )

        fft_features = self.fft_branch(frequency_inputs)

        if not self._printed_shapes:
            print(
                "[DEBUG] Fusion feature shapes | "
                f"clip={tuple(clip_features.shape)} "
                f"dino={tuple(dino_features.shape)} "
                f"efficientnet={tuple(efficientnet_features.shape)} "
                f"fft_in={tuple(frequency_inputs.shape)} "
                f"fft_feat={tuple(fft_features.shape)} "
                f"branch_dim={self.branch_feature_dim} "
                f"fusion_dim={self.fusion_input_dim}"
            )
            self._printed_shapes = True

        fused = torch.cat([clip_features, dino_features, efficientnet_features, fft_features], dim=1)
        return self.net(fused).squeeze(-1)


def build_classifier(
    clip_dim: int,
    dino_dim: int,
    efficientnet_dim: int,
    fft_dim: int = 128,
    hidden_dims: tuple[int, int] = (512, 256),
    dropout: float = 0.3,
) -> FusionMLP:
    return FusionMLP(
        clip_dim=clip_dim,
        dino_dim=dino_dim,
        efficientnet_dim=efficientnet_dim,
        fft_dim=fft_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )


def _as_tensor(array: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=torch.float32)
    return torch.as_tensor(array, dtype=torch.float32, device=device)


@torch.inference_mode()
def predict_probabilities(
    model: FusionMLP,
    clip_features: np.ndarray | torch.Tensor,
    dino_features: np.ndarray | torch.Tensor,
    efficientnet_features: np.ndarray | torch.Tensor,
    frequency_inputs: np.ndarray | torch.Tensor,
    device: str | torch.device | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(target_device)
    model.eval()

    clip_tensor = _as_tensor(clip_features, target_device)
    dino_tensor = _as_tensor(dino_features, target_device)
    efficientnet_tensor = _as_tensor(efficientnet_features, target_device)
    fft_tensor = _as_tensor(frequency_inputs, target_device)

    fake_probabilities: list[np.ndarray] = []
    for start_idx in range(0, clip_tensor.shape[0], batch_size):
        batch_clip = clip_tensor[start_idx : start_idx + batch_size]
        batch_dino = dino_tensor[start_idx : start_idx + batch_size]
        batch_eff = efficientnet_tensor[start_idx : start_idx + batch_size]
        batch_fft = fft_tensor[start_idx : start_idx + batch_size]
        logits = model(batch_clip, batch_dino, batch_eff, batch_fft)
        fake_probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        fake_probabilities.append(np.atleast_1d(fake_probs))

    fake_probabilities_array = np.concatenate(fake_probabilities, axis=0)
    return np.stack([1.0 - fake_probabilities_array, fake_probabilities_array], axis=1)


def save_fusion_model(path: str | Path, model: FusionMLP) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_version": MODEL_VERSION,
            "clip_dim": model.clip_dim,
            "dino_dim": model.dino_dim,
            "efficientnet_dim": model.efficientnet_dim,
            "branch_feature_dim": model.branch_feature_dim,
            "fft_dim": model.fft_dim,
            "fusion_input_dim": model.fusion_input_dim,
            "hidden_dims": list(model.hidden_dims),
            "dropout": model.dropout,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        path_obj,
    )


def load_fusion_model(path: str | Path, device: str | torch.device | None = None) -> FusionMLP:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {path_obj}")

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path_obj, map_location=target_device)

    if not isinstance(payload, dict) or "state_dict" not in payload or int(payload.get("model_version", 0)) < MODEL_VERSION:
        raise RuntimeError(
            "This fusion checkpoint was saved with the old output-level architecture. "
            "Please rerun train.py so artifacts match the feature-level fusion pipeline."
        )

    model = FusionMLP(
        clip_dim=int(payload["clip_dim"]),
        dino_dim=int(payload["dino_dim"]),
        efficientnet_dim=int(payload["efficientnet_dim"]),
        fft_dim=int(payload.get("fft_dim", 128)),
        hidden_dims=tuple(int(value) for value in payload.get("hidden_dims", [512, 256])),
        dropout=float(payload.get("dropout", 0.3)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(target_device)
    model.eval()
    return model
