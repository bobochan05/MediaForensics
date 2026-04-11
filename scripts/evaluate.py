from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ai.layer1_detection.models.fusion_model import load_fusion_model, predict_probabilities
from ai.shared.file_utils import LABEL_TO_NAME, load_json, load_numpy, save_json


def aggregate_video_predictions(
    frame_probabilities: np.ndarray,
    frame_video_ids: Sequence[str],
    frame_labels: Sequence[int],
    frame_video_paths: Sequence[str] | None = None,
) -> list[dict[str, float | int | str]]:
    grouped_probabilities: dict[str, list[float]] = defaultdict(list)
    grouped_labels: "OrderedDict[str, int]" = OrderedDict()
    grouped_paths: dict[str, str] = {}

    for index, video_id in enumerate(frame_video_ids):
        grouped_probabilities[video_id].append(float(frame_probabilities[index, 1]))
        grouped_labels.setdefault(video_id, int(frame_labels[index]))
        if frame_video_paths is not None:
            grouped_paths.setdefault(video_id, frame_video_paths[index])

    predictions: list[dict[str, float | int | str]] = []
    for video_id, true_label in grouped_labels.items():
        fake_probability = float(np.mean(grouped_probabilities[video_id]))
        predicted_label = int(fake_probability >= 0.5)
        confidence = fake_probability if predicted_label == 1 else 1.0 - fake_probability
        predictions.append(
            {
                "video_id": video_id,
                "video_path": grouped_paths.get(video_id, ""),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "true_name": LABEL_TO_NAME[true_label],
                "predicted_name": LABEL_TO_NAME[predicted_label],
                "fake_probability": fake_probability,
                "confidence": confidence,
            }
        )

    return predictions


def compute_video_metrics(video_predictions: Sequence[dict[str, float | int | str]]) -> dict[str, object]:
    y_true = [int(prediction["true_label"]) for prediction in video_predictions]
    y_pred = [int(prediction["predicted_label"]) for prediction in video_predictions]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[LABEL_TO_NAME[0], LABEL_TO_NAME[1]],
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
        "classification_report_text": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[LABEL_TO_NAME[0], LABEL_TO_NAME[1]],
            digits=4,
            zero_division=0,
        ),
        "num_videos": len(video_predictions),
    }


def print_metrics(metrics: dict[str, object], split_name: str) -> None:
    print(f"{split_name.capitalize()} accuracy: {metrics['accuracy']:.4f}")
    print("Confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print("Classification report:")
    print(metrics["classification_report_text"])


def load_split_artifacts(
    artifacts_dir: str | Path,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    split_dir = Path(artifacts_dir) / "embeddings" / split_name
    clip_embeddings = load_numpy(split_dir / "clip_embeddings.npy")
    dino_embeddings = load_numpy(split_dir / "dino_embeddings.npy")
    efficientnet_embeddings = load_numpy(split_dir / "efficientnet_embeddings.npy")
    frequency_inputs = load_numpy(split_dir / "frequency_inputs.npy")
    labels = load_numpy(split_dir / "labels.npy")
    frame_video_ids = list(load_json(split_dir / "frame_video_ids.json"))
    frame_video_paths = list(load_json(split_dir / "frame_video_paths.json"))
    return clip_embeddings, dino_embeddings, efficientnet_embeddings, frequency_inputs, labels, frame_video_ids, frame_video_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained deepfake detector.")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory containing model artifacts.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Which split to evaluate.")
    parser.add_argument(
        "--classifier_path",
        type=str,
        default=None,
        help="Optional path to classifier. Defaults to <artifacts_dir>/fusion_model.pth.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    classifier_path = Path(args.classifier_path) if args.classifier_path else artifacts_dir / "fusion_model.pth"

    clip_embeddings, dino_embeddings, efficientnet_embeddings, frequency_inputs, labels, frame_video_ids, frame_video_paths = load_split_artifacts(
        artifacts_dir,
        args.split,
    )
    classifier = load_fusion_model(classifier_path)
    frame_probabilities = predict_probabilities(
        classifier,
        clip_embeddings,
        dino_embeddings,
        efficientnet_embeddings,
        frequency_inputs,
    )

    video_predictions = aggregate_video_predictions(
        frame_probabilities=frame_probabilities,
        frame_video_ids=frame_video_ids,
        frame_labels=labels,
        frame_video_paths=frame_video_paths,
    )
    metrics = compute_video_metrics(video_predictions)
    print_metrics(metrics, args.split)

    output_path = artifacts_dir / f"{args.split}_evaluation.json"
    save_json(output_path, {"split": args.split, "metrics": metrics, "video_predictions": video_predictions})
    print(f"Saved evaluation to: {output_path}")


if __name__ == "__main__":
    main()
