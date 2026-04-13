from __future__ import annotations

import argparse
import os
from pathlib import Path
from threading import Lock

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ai.layer1_detection.frame_extractor import extract_sampled_frames
from ai.layer1_detection.models.clip_model import ClipModel
from ai.layer1_detection.models.dino_model import DINOModel
from ai.layer1_detection.models.efficientnet_model import EfficientNetModel
from ai.layer1_detection.models.fft_model import compute_frequency_inputs
from ai.layer1_detection.models.fusion_model import load_fusion_model, predict_probabilities
from ai.shared.video_budget import adaptive_frame_plan
from ai.shared.file_utils import LABEL_TO_NAME


_MODEL_CACHE: dict[tuple[str, str, str, str], tuple[ClipModel, EfficientNetModel, DINOModel, torch.nn.Module]] = {}
_MODEL_CACHE_LOCK = Lock()


def _resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for inference, but this Python environment cannot access it.")
    return requested_device


def _configure_torch_runtime(device: str) -> None:
    torch.set_float32_matmul_precision("high")
    if device != "cuda":
        return
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True


def _cached_model_bundle(
    *,
    classifier_path: Path,
    efficientnet_path: Path,
    dino_path: Path,
    device: str,
) -> tuple[ClipModel, EfficientNetModel, DINOModel, torch.nn.Module]:
    cache_key = (
        str(classifier_path.resolve()),
        str(efficientnet_path.resolve()),
        str(dino_path.resolve()),
        str(device),
    )
    with _MODEL_CACHE_LOCK:
        bundle = _MODEL_CACHE.get(cache_key)
        if bundle is None:
            bundle = (
                ClipModel(device=device),
                EfficientNetModel(model_path=efficientnet_path, device=device),
                DINOModel(model_path=dino_path, device=device),
                load_fusion_model(classifier_path, device=device),
            )
            _MODEL_CACHE[cache_key] = bundle
        return bundle


def predict_video(
    video_path: str | Path,
    classifier_path: str | Path = "artifacts/fusion_model.pth",
    efficientnet_path: str | Path = "artifacts/efficientnet_finetuned.pth",
    dino_path: str | Path = "artifacts/dino_finetuned.pth",
    sample_fps: float = 0.35,
    frames_per_video: int | None = None,
    image_size: int = 224,
    device: str = "auto",
) -> tuple[str, float]:
    input_path = Path(video_path)
    classifier_path = Path(classifier_path)
    efficientnet_path = Path(efficientnet_path)
    dino_path = Path(dino_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Pass a real existing path with --image_path or --video_path."
        )

    device = _resolve_device(device)
    _configure_torch_runtime(device)
    clip_model, eff_model, dino_model, classifier = _cached_model_bundle(
        classifier_path=classifier_path,
        efficientnet_path=efficientnet_path,
        dino_path=dino_path,
        device=device,
    )
    effective_sample_fps, effective_frames_per_video, _ = adaptive_frame_plan(
        input_path,
        purpose="detection",
        requested_sample_fps=sample_fps,
        requested_frames_per_video=frames_per_video,
    )

    frames = extract_sampled_frames(
        video_path=input_path,
        image_size=image_size,
        sample_fps=effective_sample_fps,
        frames_per_video=effective_frames_per_video,
        purpose="detection",
    )
    if not frames:
        raise ValueError(f"No frames could be extracted from {input_path}")

    clip_features = clip_model.extract_features(frames)
    dino_features = dino_model.extract_features(frames)
    efficientnet_features = eff_model.extract_features(frames)
    frequency_inputs = compute_frequency_inputs(frames, image_size=image_size).astype(np.float32)

    frame_probabilities = predict_probabilities(
        classifier,
        clip_features,
        dino_features,
        efficientnet_features,
        frequency_inputs,
        device=device,
    )
    fake_probability = float(np.mean(frame_probabilities[:, 1]))
    predicted_label = int(fake_probability >= 0.5)
    confidence = fake_probability if predicted_label == 1 else 1.0 - fake_probability

    return LABEL_TO_NAME[predicted_label], confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deepfake inference on a single image or video.")
    parser.add_argument("--video_path", type=str, default=None, help="Path to the input video.")
    parser.add_argument("--image_path", type=str, default=None, help="Path to the input image.")
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Generic input path alias for either an image or a video.",
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        default="artifacts/fusion_model.pth",
        help="Path to the saved classifier.",
    )
    parser.add_argument(
        "--efficientnet_path",
        type=str,
        default="artifacts/efficientnet_finetuned.pth",
        help="Path to the fine-tuned EfficientNet weights.",
    )
    parser.add_argument(
        "--dino_path",
        type=str,
        default="artifacts/dino_finetuned.pth",
        help="Path to the fine-tuned DINO weights.",
    )
    parser.add_argument("--sample_fps", type=float, default=0.35, help="Frames per second to sample.")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"), help="Inference device.")
    parser.add_argument(
        "--frames_per_video",
        type=int,
        default=None,
        help="Optional fixed number of frames to sample instead of FPS-based sampling.",
    )
    parser.add_argument("--image_size", type=int, default=224, help="Square image size for resized frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path or args.image_path or args.video_path
    if input_path is None:
        raise ValueError("Provide --input_path, --image_path, or --video_path")

    label, confidence = predict_video(
        video_path=input_path,
        classifier_path=args.classifier_path,
        efficientnet_path=args.efficientnet_path,
        dino_path=args.dino_path,
        sample_fps=args.sample_fps,
        frames_per_video=args.frames_per_video,
        image_size=args.image_size,
        device=args.device,
    )
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
