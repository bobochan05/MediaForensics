from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


@dataclass
class AudioFeatureBundle:
    fft_bins: np.ndarray
    mel_spectrogram: np.ndarray
    mfcc: np.ndarray
    summary_vector: np.ndarray


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor - tensor.amin()
    denom = torch.clamp(tensor.amax(), min=1e-6)
    return tensor / denom


def compute_audio_feature_bundle(
    waveform: torch.Tensor,
    sample_rate: int,
    max_duration_seconds: float = 15.0,
) -> AudioFeatureBundle:
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.to(torch.float32).flatten()
    max_samples = int(sample_rate * max_duration_seconds)
    if waveform.numel() > max_samples:
        start = max((waveform.numel() - max_samples) // 2, 0)
        waveform = waveform[start : start + max_samples]

    fft = torch.fft.rfft(waveform)
    fft_magnitude = torch.log1p(torch.abs(fft))
    fft_magnitude = F.adaptive_avg_pool1d(fft_magnitude.view(1, 1, -1), 256).view(-1)
    fft_magnitude = _normalize_tensor(fft_magnitude)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
    )
    mel = torch.log1p(mel_transform(waveform.unsqueeze(0)))
    mel = F.adaptive_avg_pool1d(mel, 128)
    mel = _normalize_tensor(mel)

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=40,
        melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 64},
    )
    mfcc = mfcc_transform(waveform.unsqueeze(0))
    mfcc = F.adaptive_avg_pool1d(mfcc, 128)
    mfcc = _normalize_tensor(mfcc)

    summary_vector = torch.cat(
        [
            fft_magnitude,
            mel.mean(dim=-1).view(-1),
            mel.std(dim=-1).view(-1),
            mfcc.mean(dim=-1).view(-1),
            mfcc.std(dim=-1).view(-1),
        ],
        dim=0,
    )
    summary_vector = summary_vector / torch.clamp(torch.linalg.norm(summary_vector), min=1e-6)

    return AudioFeatureBundle(
        fft_bins=fft_magnitude.cpu().numpy().astype(np.float32),
        mel_spectrogram=mel.squeeze(0).cpu().numpy().astype(np.float32),
        mfcc=mfcc.squeeze(0).cpu().numpy().astype(np.float32),
        summary_vector=summary_vector.cpu().numpy().astype(np.float32),
    )

