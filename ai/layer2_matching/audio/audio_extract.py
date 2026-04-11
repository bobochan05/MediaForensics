from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import av
import torch
import torchaudio

from ai.shared.file_utils import ensure_dir

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class AudioExtractionResult:
    has_audio: bool
    waveform: torch.Tensor | None = None
    sample_rate: int | None = None
    extracted_path: str | None = None
    duration_seconds: float = 0.0


def _to_mono_resampled(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int) -> tuple[torch.Tensor, int]:
    waveform = waveform.to(torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    peak = float(waveform.abs().max() or 1.0)
    if peak > 1.5:
        waveform = waveform / peak
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return waveform.contiguous(), sample_rate


def _extract_with_ffmpeg(video_path: Path, output_path: Path, target_sample_rate: int) -> tuple[torch.Tensor, int] | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not output_path.exists():
        return None

    waveform, sample_rate = torchaudio.load(str(output_path))
    return waveform, sample_rate


def _extract_with_pyav(video_path: Path) -> tuple[torch.Tensor, int] | None:
    container = av.open(str(video_path))
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return None

    chunks: list[torch.Tensor] = []
    sample_rate = 0
    for frame in container.decode(audio=0):
        sample_rate = int(frame.sample_rate or sample_rate)
        array = frame.to_ndarray()
        tensor = torch.from_numpy(array)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 2 and tensor.shape[0] > 8 and tensor.shape[1] <= 8:
            tensor = tensor.transpose(0, 1)
        chunks.append(tensor.to(torch.float32))

    if not chunks or sample_rate <= 0:
        return None

    waveform = torch.cat(chunks, dim=-1)
    return waveform, sample_rate


def extract_audio_from_media(
    media_path: str | Path,
    output_dir: str | Path,
    target_sample_rate: int = 16000,
) -> AudioExtractionResult:
    media_path = Path(media_path)
    output_dir = ensure_dir(output_dir)
    output_path = output_dir / f"{media_path.stem}_audio.wav"

    if media_path.suffix.lower() in AUDIO_EXTENSIONS:
        waveform, sample_rate = torchaudio.load(str(media_path))
    elif media_path.suffix.lower() in VIDEO_EXTENSIONS:
        extracted = _extract_with_ffmpeg(media_path, output_path, target_sample_rate)
        if extracted is None:
            extracted = _extract_with_pyav(media_path)
        if extracted is None:
            return AudioExtractionResult(has_audio=False)
        waveform, sample_rate = extracted
    else:
        return AudioExtractionResult(has_audio=False)

    waveform, sample_rate = _to_mono_resampled(waveform, int(sample_rate), target_sample_rate)
    duration_seconds = float(waveform.shape[-1] / max(sample_rate, 1))
    torchaudio.save(str(output_path), waveform.cpu(), sample_rate)
    return AudioExtractionResult(
        has_audio=bool(waveform.numel() > 0),
        waveform=waveform,
        sample_rate=sample_rate,
        extracted_path=str(output_path),
        duration_seconds=duration_seconds,
    )

