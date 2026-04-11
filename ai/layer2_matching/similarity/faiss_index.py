from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return (vectors / norms).astype(np.float32)


class FaissVectorIndex:
    def __init__(self, name: str, storage_dir: str | Path) -> None:
        self.name = name
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_dir / f"{name}.index"
        self.metadata_path = self.storage_dir / f"{name}_metadata.json"
        self.index: faiss.IndexFlatIP | None = None
        self.dimension: int | None = None
        self.metadata: list[dict[str, object]] = []
        self.load()

    @property
    def size(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    def load(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.dimension = int(self.index.d)
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def _ensure_index(self, dimension: int) -> None:
        if self.index is None:
            self.dimension = int(dimension)
            self.index = faiss.IndexFlatIP(self.dimension)
            return
        if self.dimension != int(dimension):
            raise ValueError(f"Index '{self.name}' expects dimension {self.dimension}, got {dimension}.")

    def add(self, vectors: np.ndarray, metadata: list[dict[str, object]]) -> None:
        vectors = normalize_rows(vectors)
        if len(metadata) != len(vectors):
            raise ValueError("Vector count and metadata count must match.")
        if len(vectors) == 0:
            return

        self._ensure_index(vectors.shape[1])
        assert self.index is not None
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query: np.ndarray, top_k: int = 5) -> list[dict[str, object]]:
        if self.index is None or self.size == 0:
            return []

        query = normalize_rows(query)
        top_k = min(max(int(top_k), 1), self.size)
        scores, indices = self.index.search(query, top_k)
        results: list[dict[str, object]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append(
                {
                    "score": float(score),
                    "metadata": self.metadata[idx],
                }
            )
        return results

    def save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

