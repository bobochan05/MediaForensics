from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn


def compute_frequency_inputs(images: Sequence[Image.Image], image_size: int) -> np.ndarray:
    grayscale_tensors: list[torch.Tensor] = []
    for image in images:
        grayscale = image.convert("L").resize((image_size, image_size))
        pixel_tensor = torch.from_numpy(np.asarray(grayscale, dtype=np.float32) / 255.0)
        grayscale_tensors.append(pixel_tensor)

    if not grayscale_tensors:
        return np.empty((0, 1, image_size, image_size), dtype=np.float32)

    batch = torch.stack(grayscale_tensors, dim=0)
    fft_map = torch.fft.fft2(batch)
    magnitude = torch.log1p(torch.abs(torch.fft.fftshift(fft_map, dim=(-2, -1))))

    min_values = magnitude.amin(dim=(-2, -1), keepdim=True)
    max_values = magnitude.amax(dim=(-2, -1), keepdim=True)
    magnitude = magnitude - min_values
    magnitude = magnitude / torch.clamp(max_values - min_values, min=1e-6)

    return magnitude.unsqueeze(1).cpu().numpy().astype(np.float32)


class FFTBranch(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, frequency_inputs: torch.Tensor) -> torch.Tensor:
        if frequency_inputs.ndim == 3:
            frequency_inputs = frequency_inputs.unsqueeze(1)
        return self.net(frequency_inputs)
