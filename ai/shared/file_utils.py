from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

LABEL_TO_NAME = {0: "real", 1: "fake"}


@dataclass
class SplitEmbeddings:
    clip_embeddings: np.ndarray
    dino_embeddings: np.ndarray
    efficientnet_embeddings: np.ndarray
    frequency_inputs: np.ndarray
    frame_labels: np.ndarray
    frame_video_ids: list[str]
    frame_video_paths: list[str]


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_numpy(path: str | Path, array: np.ndarray) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    np.save(path_obj, array)


def load_numpy(path: str | Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def save_model(path: str | Path, model: Any) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    joblib.dump(model, path_obj)


def load_model(path: str | Path) -> Any:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {path_obj}")
    return joblib.load(path_obj)


def save_json(path: str | Path, payload: Any) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
