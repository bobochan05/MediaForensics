from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from ai.layer1_detection.data_loader import FAKE_LABEL, REAL_LABEL, VideoRecord, collect_faceforensics_videos, summarize_records
from scripts.evaluate import aggregate_video_predictions, compute_video_metrics, print_metrics
from ai.layer1_detection.frame_extractor import extract_sampled_frames
from ai.layer1_detection.models.clip_model import ClipModel
from ai.layer1_detection.models.dino_model import DINOModel
from ai.layer1_detection.models.efficientnet_model import EfficientNetModel
from ai.layer1_detection.models.fft_model import compute_frequency_inputs
from ai.layer1_detection.models.fusion_model import build_classifier, predict_probabilities, save_fusion_model
from ai.shared.file_utils import SplitEmbeddings, configure_logging, ensure_dir, load_json, load_numpy, save_json, save_numpy, set_seed
from ai.layer1_detection.video_split import save_split_manifest, split_video_records

LOGGER = logging.getLogger(__name__)


@dataclass
class ExtractedFrames:
    images: list[Image.Image]
    labels: np.ndarray
    frame_video_ids: list[str]
    frame_video_paths: list[str]


class ImageListDataset(Dataset):
    def __init__(self, images: list[Image.Image], labels: np.ndarray, transform):
        self.images = images
        self.labels = labels.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.transform(self.images[index].convert("RGB"))
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return image, label


class FeatureProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for training, but this Python environment cannot access it. "
            "Run training with the project virtualenv at '.venv\\Scripts\\python.exe'."
        )

    return requested_device


def _configure_torch_runtime(device: str) -> None:
    torch.set_float32_matmul_precision("high")
    if device != "cuda":
        LOGGER.info("Using device: cpu")
        return

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True

    LOGGER.info("Using device: cuda | gpu=%s", torch.cuda.get_device_name(0))


def _loader_kwargs(device: str) -> dict[str, object]:
    return {"pin_memory": device == "cuda"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deepfake detector with feature-level fusion.")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--frames_per_video", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_real_videos", type=int)
    parser.add_argument("--max_fake_videos", type=int)
    parser.add_argument("--reuse_embeddings", action="store_true")
    parser.add_argument("--epochs", type=int, default=15, help="Feature-fusion training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Feature-fusion learning rate.")
    parser.add_argument("--backbone_epochs", type=int, default=15, help="Fine-tuning epochs for EfficientNet and DINO.")
    parser.add_argument("--backbone_learning_rate", type=float, default=5e-5, help="Learning rate for EfficientNet and DINO fine-tuning.")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for fine-tuning optimizers.")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Binary label smoothing used for branch and fusion training.")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--device", type=str, default="cuda", choices=("auto", "cuda", "cpu"), help="Training device. Default keeps training off CPU-only environments.")
    return parser.parse_args()


def balance_video_records(video_records: list[VideoRecord], random_state: int) -> list[VideoRecord]:
    real_records = [record for record in video_records if record.label == REAL_LABEL]
    fake_records = [record for record in video_records if record.label == FAKE_LABEL]

    if not real_records or not fake_records:
        raise ValueError("Both real and fake samples are required for balanced training.")

    target_count = min(len(real_records), len(fake_records))
    rng = np.random.default_rng(random_state)
    real_indices = sorted(rng.choice(len(real_records), size=target_count, replace=False).tolist())
    fake_indices = sorted(rng.choice(len(fake_records), size=target_count, replace=False).tolist())

    balanced_records = [real_records[index] for index in real_indices] + [fake_records[index] for index in fake_indices]
    rng.shuffle(balanced_records)
    LOGGER.info(
        "Balanced dataset from real=%s fake=%s to real=%s fake=%s",
        len(real_records),
        len(fake_records),
        target_count,
        target_count,
    )
    return balanced_records


def extract_split_frames(
    split_name: str,
    split_records: list[VideoRecord],
    sample_fps: float,
    frames_per_video: int | None,
    image_size: int,
) -> ExtractedFrames:
    images: list[Image.Image] = []
    labels: list[float] = []
    frame_video_ids: list[str] = []
    frame_video_paths: list[str] = []

    for record in tqdm(split_records, desc=f"Extracting {split_name} frames"):
        frames = extract_sampled_frames(
            video_path=record.video_path,
            image_size=image_size,
            sample_fps=sample_fps,
            frames_per_video=frames_per_video,
        )
        if not frames:
            LOGGER.warning("Skipping input with no extracted frames: %s", record.video_path)
            continue

        images.extend(frames)
        labels.extend(float(record.label) for _ in frames)
        frame_video_ids.extend(record.video_id for _ in frames)
        frame_video_paths.extend(str(record.video_path) for _ in frames)

    if not images:
        raise ValueError(f"No frames were extracted for split '{split_name}'.")

    return ExtractedFrames(
        images=images,
        labels=np.asarray(labels, dtype=np.float32),
        frame_video_ids=frame_video_ids,
        frame_video_paths=frame_video_paths,
    )


def _build_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(labels.astype(np.int64), minlength=2)
    if np.any(class_counts == 0):
        raise ValueError(f"Both classes are required, got counts={class_counts.tolist()}")

    sample_weights = np.where(labels == REAL_LABEL, 1.0 / class_counts[REAL_LABEL], 1.0 / class_counts[FAKE_LABEL])
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def smooth_binary_labels(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return labels
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def fine_tune_binary_model(
    model_name: str,
    model_wrapper,
    images: list[Image.Image],
    labels: np.ndarray,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    gradient_clip_norm: float,
    device: str,
) -> None:
    sampler = _build_balanced_sampler(labels)
    dataset = ImageListDataset(images, labels, model_wrapper.train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, **_loader_kwargs(device))

    probe_head = FeatureProbe(model_wrapper.feature_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()

    dino_freeze_epochs = max(1, epochs // 3) if model_name == "DINO" and epochs > 1 else 0
    if model_name == "DINO":
        for parameter in model_wrapper.parameters():
            parameter.requires_grad = False
    else:
        model_wrapper.unfreeze_last_layers()

    def _build_optimizer():
        params = list(probe_head.parameters()) + list(model_wrapper.trainable_parameters())
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    optimizer = _build_optimizer()

    for epoch in range(epochs):
        if model_name == "DINO" and epoch == dino_freeze_epochs:
            model_wrapper.unfreeze_last_layers()
            optimizer = _build_optimizer()
            LOGGER.info("Unfroze DINO last layers after %s warmup epochs", dino_freeze_epochs)

        model_wrapper.train_mode()
        probe_head.train()
        total_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=device == "cuda")
            batch_y = smooth_binary_labels(batch_y.to(device, non_blocking=device == "cuda"), label_smoothing)

            features = model_wrapper.extract_features_from_tensors(batch_x)
            logits = probe_head(features)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(probe_head.parameters()) + list(model_wrapper.trainable_parameters()), max_norm=gradient_clip_norm)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        LOGGER.info(
            "%s epoch %s/%s | loss=%.4f",
            model_name,
            epoch + 1,
            epochs,
            total_loss / max(len(dataset), 1),
        )

    model_wrapper.eval_mode()


def prepare_split_embeddings(
    split_name: str,
    extracted_frames: ExtractedFrames,
    clip_model: ClipModel,
    eff_model: EfficientNetModel,
    dino_model: DINOModel,
    output_dir: Path,
    batch_size: int,
    image_size: int,
    reuse_embeddings: bool,
    cache_metadata: dict[str, object],
) -> SplitEmbeddings:
    split_dir = ensure_dir(output_dir / "embeddings" / split_name)
    clip_embeddings_path = split_dir / "clip_embeddings.npy"
    dino_embeddings_path = split_dir / "dino_embeddings.npy"
    efficientnet_embeddings_path = split_dir / "efficientnet_embeddings.npy"
    frequency_inputs_path = split_dir / "frequency_inputs.npy"
    labels_path = split_dir / "labels.npy"
    frame_video_ids_path = split_dir / "frame_video_ids.json"
    frame_video_paths_path = split_dir / "frame_video_paths.json"
    metadata_path = split_dir / "metadata.json"

    required_paths = [
        clip_embeddings_path,
        dino_embeddings_path,
        efficientnet_embeddings_path,
        frequency_inputs_path,
        labels_path,
        frame_video_ids_path,
        frame_video_paths_path,
    ]

    if reuse_embeddings and metadata_path.exists() and all(path.exists() for path in required_paths):
        cached_metadata = load_json(metadata_path)
        if cached_metadata == cache_metadata:
            LOGGER.info("Loading cached %s feature tensors", split_name)
            return SplitEmbeddings(
                clip_embeddings=load_numpy(clip_embeddings_path),
                dino_embeddings=load_numpy(dino_embeddings_path),
                efficientnet_embeddings=load_numpy(efficientnet_embeddings_path),
                frequency_inputs=load_numpy(frequency_inputs_path),
                frame_labels=load_numpy(labels_path),
                frame_video_ids=list(load_json(frame_video_ids_path)),
                frame_video_paths=list(load_json(frame_video_paths_path)),
            )

    LOGGER.info("Extracting %s branch features", split_name)
    clip_batches: list[np.ndarray] = []
    dino_batches: list[np.ndarray] = []
    efficientnet_batches: list[np.ndarray] = []
    frequency_batches: list[np.ndarray] = []

    for start_idx in tqdm(range(0, len(extracted_frames.images), batch_size), desc=f"Encoding {split_name}"):
        batch_images = extracted_frames.images[start_idx : start_idx + batch_size]
        clip_features = clip_model.extract_features(batch_images)
        dino_features = dino_model.extract_features(batch_images)
        efficientnet_features = eff_model.extract_features(batch_images)
        fft_inputs = compute_frequency_inputs(batch_images, image_size=image_size)

        clip_batches.append(clip_features.astype(np.float32))
        dino_batches.append(dino_features.astype(np.float32))
        efficientnet_batches.append(efficientnet_features.astype(np.float32))
        frequency_batches.append(fft_inputs.astype(np.float32))

    split_embeddings = SplitEmbeddings(
        clip_embeddings=np.concatenate(clip_batches, axis=0).astype(np.float32),
        dino_embeddings=np.concatenate(dino_batches, axis=0).astype(np.float32),
        efficientnet_embeddings=np.concatenate(efficientnet_batches, axis=0).astype(np.float32),
        frequency_inputs=np.concatenate(frequency_batches, axis=0).astype(np.float32),
        frame_labels=extracted_frames.labels.astype(np.float32),
        frame_video_ids=extracted_frames.frame_video_ids,
        frame_video_paths=extracted_frames.frame_video_paths,
    )

    save_numpy(clip_embeddings_path, split_embeddings.clip_embeddings)
    save_numpy(dino_embeddings_path, split_embeddings.dino_embeddings)
    save_numpy(efficientnet_embeddings_path, split_embeddings.efficientnet_embeddings)
    save_numpy(frequency_inputs_path, split_embeddings.frequency_inputs)
    save_numpy(labels_path, split_embeddings.frame_labels)
    save_json(frame_video_ids_path, split_embeddings.frame_video_ids)
    save_json(frame_video_paths_path, split_embeddings.frame_video_paths)
    save_json(metadata_path, cache_metadata)
    return split_embeddings


def train_classifier(
    classifier: nn.Module,
    clip_embeddings: np.ndarray,
    dino_embeddings: np.ndarray,
    efficientnet_embeddings: np.ndarray,
    frequency_inputs: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    epochs: int,
    lr: float,
    label_smoothing: float,
    device: str,
) -> None:
    sampler = _build_balanced_sampler(labels)
    dataset = TensorDataset(
        torch.tensor(clip_embeddings, dtype=torch.float32),
        torch.tensor(dino_embeddings, dtype=torch.float32),
        torch.tensor(efficientnet_embeddings, dtype=torch.float32),
        torch.tensor(frequency_inputs, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, **_loader_kwargs(device))

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    classifier.to(device)

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0

        for batch_clip, batch_dino, batch_eff, batch_fft, batch_y in loader:
            batch_clip = batch_clip.to(device, non_blocking=device == "cuda")
            batch_dino = batch_dino.to(device, non_blocking=device == "cuda")
            batch_eff = batch_eff.to(device, non_blocking=device == "cuda")
            batch_fft = batch_fft.to(device, non_blocking=device == "cuda")
            batch_y = smooth_binary_labels(batch_y.to(device, non_blocking=device == "cuda"), label_smoothing)

            logits = classifier(batch_clip, batch_dino, batch_eff, batch_fft)
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_clip.size(0)

        LOGGER.info("Fusion epoch %s/%s | loss=%.4f", epoch + 1, epochs, total_loss / max(len(dataset), 1))


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.random_state)
    device = _resolve_device(args.device)
    _configure_torch_runtime(device)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Invalid dataset_dir: {dataset_dir}")

    output_dir = ensure_dir(args.output_dir)
    video_records = collect_faceforensics_videos(
        dataset_dir=args.dataset_dir,
        max_real_videos=args.max_real_videos,
        max_fake_videos=args.max_fake_videos,
        random_state=args.random_state,
    )
    original_dataset_summary = summarize_records(video_records)
    video_records = balance_video_records(video_records, random_state=args.random_state)
    balanced_dataset_summary = summarize_records(video_records)

    train_records, test_records = split_video_records(
        video_records,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    save_split_manifest(output_dir / "split_manifest.json", train_records, test_records)

    train_frames = extract_split_frames(
        split_name="train",
        split_records=train_records,
        sample_fps=args.sample_fps,
        frames_per_video=args.frames_per_video,
        image_size=args.image_size,
    )
    test_frames = extract_split_frames(
        split_name="test",
        split_records=test_records,
        sample_fps=args.sample_fps,
        frames_per_video=args.frames_per_video,
        image_size=args.image_size,
    )

    clip_model = ClipModel(device=device)
    eff_model = EfficientNetModel(device=device)
    dino_model = DINOModel(device=device)

    fine_tune_binary_model(
        model_name="EfficientNet",
        model_wrapper=eff_model,
        images=train_frames.images,
        labels=train_frames.labels,
        batch_size=args.batch_size,
        epochs=args.backbone_epochs,
        learning_rate=args.backbone_learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        gradient_clip_norm=args.gradient_clip_norm,
        device=device,
    )
    fine_tune_binary_model(
        model_name="DINO",
        model_wrapper=dino_model,
        images=train_frames.images,
        labels=train_frames.labels,
        batch_size=args.batch_size,
        epochs=args.backbone_epochs,
        learning_rate=args.backbone_learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        gradient_clip_norm=args.gradient_clip_norm,
        device=device,
    )

    efficientnet_path = output_dir / "efficientnet_finetuned.pth"
    dino_path = output_dir / "dino_finetuned.pth"
    eff_model.save(efficientnet_path)
    dino_model.save(dino_path)

    cache_metadata = {
        "fusion_mode": "feature_level",
        "sample_fps": args.sample_fps,
        "frames_per_video": args.frames_per_video,
        "image_size": args.image_size,
        "num_train_frames": len(train_frames.images),
        "num_test_frames": len(test_frames.images),
        "efficientnet_path": str(efficientnet_path),
        "efficientnet_mtime": efficientnet_path.stat().st_mtime,
        "dino_path": str(dino_path),
        "dino_mtime": dino_path.stat().st_mtime,
    }

    train_split = prepare_split_embeddings(
        split_name="train",
        extracted_frames=train_frames,
        clip_model=clip_model,
        eff_model=eff_model,
        dino_model=dino_model,
        output_dir=output_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        reuse_embeddings=args.reuse_embeddings,
        cache_metadata=cache_metadata,
    )
    test_split = prepare_split_embeddings(
        split_name="test",
        extracted_frames=test_frames,
        clip_model=clip_model,
        eff_model=eff_model,
        dino_model=dino_model,
        output_dir=output_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        reuse_embeddings=args.reuse_embeddings,
        cache_metadata=cache_metadata,
    )

    classifier = build_classifier(
        clip_dim=train_split.clip_embeddings.shape[1],
        dino_dim=train_split.dino_embeddings.shape[1],
        efficientnet_dim=train_split.efficientnet_embeddings.shape[1],
    )
    LOGGER.info(
        "Fusion branch feature dim=%s (clip=%s + dino=%s + efficientnet=%s), total fusion input dim=%s including fft=%s",
        classifier.branch_feature_dim,
        train_split.clip_embeddings.shape[1],
        train_split.dino_embeddings.shape[1],
        train_split.efficientnet_embeddings.shape[1],
        classifier.fusion_input_dim,
        classifier.fft_dim,
    )
    train_classifier(
        classifier=classifier,
        clip_embeddings=train_split.clip_embeddings,
        dino_embeddings=train_split.dino_embeddings,
        efficientnet_embeddings=train_split.efficientnet_embeddings,
        frequency_inputs=train_split.frequency_inputs,
        labels=train_split.frame_labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        label_smoothing=args.label_smoothing,
        device=device,
    )
    save_fusion_model(output_dir / "fusion_model.pth", classifier)

    train_frame_probabilities = predict_probabilities(
        classifier,
        train_split.clip_embeddings,
        train_split.dino_embeddings,
        train_split.efficientnet_embeddings,
        train_split.frequency_inputs,
        device=device,
    )
    test_frame_probabilities = predict_probabilities(
        classifier,
        test_split.clip_embeddings,
        test_split.dino_embeddings,
        test_split.efficientnet_embeddings,
        test_split.frequency_inputs,
        device=device,
    )

    train_video_predictions = aggregate_video_predictions(
        frame_probabilities=train_frame_probabilities,
        frame_video_ids=train_split.frame_video_ids,
        frame_labels=train_split.frame_labels,
        frame_video_paths=train_split.frame_video_paths,
    )
    test_video_predictions = aggregate_video_predictions(
        frame_probabilities=test_frame_probabilities,
        frame_video_ids=test_split.frame_video_ids,
        frame_labels=test_split.frame_labels,
        frame_video_paths=test_split.frame_video_paths,
    )

    train_metrics = compute_video_metrics(train_video_predictions)
    test_metrics = compute_video_metrics(test_video_predictions)
    print_metrics(train_metrics, "train")
    print_metrics(test_metrics, "test")

    save_json(
        output_dir / "metrics.json",
        {
            "dataset_summary_before_balancing": original_dataset_summary,
            "dataset_summary_after_balancing": balanced_dataset_summary,
            "train_samples": len(train_records),
            "test_samples": len(test_records),
            "train_frames": len(train_frames.images),
            "test_frames": len(test_frames.images),
            "clip_feature_dim": int(train_split.clip_embeddings.shape[1]),
            "dino_feature_dim": int(train_split.dino_embeddings.shape[1]),
            "efficientnet_feature_dim": int(train_split.efficientnet_embeddings.shape[1]),
            "branch_feature_dim": int(classifier.branch_feature_dim),
            "fusion_input_dim": int(classifier.fusion_input_dim),
            "frequency_input_shape": list(train_split.frequency_inputs.shape[1:]),
            "efficientnet_path": str(efficientnet_path),
            "dino_path": str(dino_path),
            "fusion_path": str(output_dir / "fusion_model.pth"),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        },
    )
    save_json(output_dir / "train_video_predictions.json", train_video_predictions)
    save_json(output_dir / "test_video_predictions.json", test_video_predictions)

    print(f"Saved EfficientNet to: {efficientnet_path}")
    print(f"Saved DINO to: {dino_path}")
    print(f"Saved fusion model to: {output_dir / 'fusion_model.pth'}")
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
