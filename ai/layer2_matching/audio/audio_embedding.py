from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from transformers import ClapModel, ClapProcessor, Wav2Vec2Model, Wav2Vec2Processor

from ai.layer2_matching.audio.audio_features import AudioFeatureBundle, compute_audio_feature_bundle


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


@dataclass
class AudioEmbeddingResult:
    has_audio: bool
    combined_embedding: np.ndarray | None = None
    wav2vec_embedding: np.ndarray | None = None
    clap_embedding: np.ndarray | None = None
    feature_bundle: AudioFeatureBundle | None = None
    sample_rate: int | None = None
    duration_seconds: float = 0.0


class AudioEmbeddingService:
    def __init__(
        self,
        device: str = "auto",
        wav2vec_model_name: str = "facebook/wav2vec2-base-960h",
        clap_model_name: str = "laion/clap-htsat-unfused",
    ) -> None:
        self.requested_device = device
        self.wav2vec_model_name = wav2vec_model_name
        self.clap_model_name = clap_model_name
        self._device: str | None = None
        self._wav2vec_processor: Wav2Vec2Processor | None = None
        self._wav2vec_model: Wav2Vec2Model | None = None
        self._clap_processor: ClapProcessor | None = None
        self._clap_model: ClapModel | None = None

    def _resolve_device(self) -> str:
        if self.requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.requested_device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.requested_device

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    def _load_wav2vec(self) -> tuple[Wav2Vec2Processor, Wav2Vec2Model]:
        if self._wav2vec_processor is None or self._wav2vec_model is None:
            self._wav2vec_processor = self._load_wav2vec_processor()
            self._wav2vec_model = self._load_wav2vec_model().to(self.device)
            self._wav2vec_model.eval()
        return self._wav2vec_processor, self._wav2vec_model

    def _load_clap(self) -> tuple[ClapProcessor, ClapModel]:
        if self._clap_processor is None or self._clap_model is None:
            self._clap_processor = self._load_clap_processor()
            self._clap_model = self._load_clap_model().to(self.device)
            self._clap_model.eval()
        return self._clap_processor, self._clap_model

    def _load_wav2vec_processor(self) -> Wav2Vec2Processor:
        try:
            return Wav2Vec2Processor.from_pretrained(self.wav2vec_model_name, local_files_only=True)
        except OSError:
            if os.getenv("DEEPFAKE_ALLOW_MODEL_DOWNLOADS") == "1":
                return Wav2Vec2Processor.from_pretrained(self.wav2vec_model_name)
            raise

    def _load_wav2vec_model(self) -> Wav2Vec2Model:
        try:
            return Wav2Vec2Model.from_pretrained(
                self.wav2vec_model_name,
                local_files_only=True,
                use_safetensors=False,
            )
        except OSError:
            if os.getenv("DEEPFAKE_ALLOW_MODEL_DOWNLOADS") == "1":
                return Wav2Vec2Model.from_pretrained(self.wav2vec_model_name, use_safetensors=False)
            raise

    def _load_clap_processor(self) -> ClapProcessor:
        try:
            return ClapProcessor.from_pretrained(self.clap_model_name, local_files_only=True)
        except OSError:
            if os.getenv("DEEPFAKE_ALLOW_MODEL_DOWNLOADS") == "1":
                return ClapProcessor.from_pretrained(self.clap_model_name)
            raise

    def _load_clap_model(self) -> ClapModel:
        try:
            return ClapModel.from_pretrained(
                self.clap_model_name,
                local_files_only=True,
                use_safetensors=False,
            )
        except OSError:
            if os.getenv("DEEPFAKE_ALLOW_MODEL_DOWNLOADS") == "1":
                return ClapModel.from_pretrained(self.clap_model_name, use_safetensors=False)
            raise

    @staticmethod
    def _time_warp(waveform: torch.Tensor, factor: float) -> torch.Tensor:
        length = waveform.shape[-1]
        warped_length = max(int(length / factor), 1)
        warped = F.interpolate(waveform.unsqueeze(0), size=warped_length, mode="linear", align_corners=False).squeeze(0)
        if warped.shape[-1] < length:
            warped = F.pad(warped, (0, length - warped.shape[-1]))
        return warped[..., :length]

    def _build_views(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        waveform = waveform.to(torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return [
            waveform,
            self._time_warp(waveform, 0.97),
            self._time_warp(waveform, 1.03),
        ]

    def _embed_wav2vec(self, waveform: torch.Tensor) -> np.ndarray:
        processor, model = self._load_wav2vec()
        views_16k = self._build_views(waveform)
        inputs = processor(
            [view.squeeze(0).cpu().numpy() for view in views_16k],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        tensor_inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**tensor_inputs)
        pooled = outputs.last_hidden_state.mean(dim=1)
        embedding = pooled.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        return _normalize(embedding)

    def _embed_clap(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        processor, model = self._load_clap()
        waveform_48k = waveform
        if sample_rate != 48000:
            waveform_48k = torchaudio.functional.resample(waveform, sample_rate, 48000)

        views_48k = self._build_views(waveform_48k)
        inputs = processor(
            audio=[view.squeeze(0).cpu().numpy() for view in views_48k],
            sampling_rate=48000,
            return_tensors="pt",
        )
        tensor_inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model.get_audio_features(**tensor_inputs)

        if isinstance(outputs, tuple):
            features = outputs[0]
        elif isinstance(outputs, torch.Tensor):
            features = outputs
        elif hasattr(outputs, "audio_embeds") and outputs.audio_embeds is not None:
            features = outputs.audio_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            raise TypeError(f"Unsupported CLAP output type: {type(outputs)!r}")
        embedding = features.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        return _normalize(embedding)

    def embed_audio(
        self,
        waveform: torch.Tensor | None,
        sample_rate: int | None,
        duration_seconds: float = 0.0,
    ) -> AudioEmbeddingResult:
        if waveform is None or sample_rate is None or waveform.numel() == 0:
            return AudioEmbeddingResult(has_audio=False)

        feature_bundle = compute_audio_feature_bundle(waveform, int(sample_rate))
        wav2vec_embedding: np.ndarray | None = None
        clap_embedding: np.ndarray | None = None

        try:
            wav2vec_embedding = self._embed_wav2vec(waveform)
        except Exception:
            wav2vec_embedding = None

        try:
            clap_embedding = self._embed_clap(waveform, int(sample_rate))
        except Exception:
            clap_embedding = None

        combined_parts = [feature_bundle.summary_vector]
        if wav2vec_embedding is not None:
            combined_parts.insert(0, wav2vec_embedding)
        if clap_embedding is not None:
            combined_parts.insert(1 if wav2vec_embedding is not None else 0, clap_embedding)

        combined = np.concatenate(combined_parts, axis=0).astype(np.float32)
        combined = _normalize(combined)

        return AudioEmbeddingResult(
            has_audio=True,
            combined_embedding=combined,
            wav2vec_embedding=wav2vec_embedding,
            clap_embedding=clap_embedding,
            feature_bundle=feature_bundle,
            sample_rate=int(sample_rate),
            duration_seconds=float(duration_seconds),
        )
