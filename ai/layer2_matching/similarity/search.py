from __future__ import annotations

from typing import Any

import numpy as np

from ai.layer2_matching.similarity.faiss_index import FaissVectorIndex


class MultimodalSimilaritySearch:
    def __init__(
        self,
        visual_index: FaissVectorIndex,
        audio_index: FaissVectorIndex,
        visual_weight: float = 0.65,
        audio_weight: float = 0.35,
    ) -> None:
        self.visual_index = visual_index
        self.audio_index = audio_index
        self.visual_weight = float(visual_weight)
        self.audio_weight = float(audio_weight)

    def search(
        self,
        visual_embedding: np.ndarray,
        audio_embedding: np.ndarray | None = None,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        visual_hits = self.visual_index.search(visual_embedding, top_k=top_k)
        audio_hits = self.audio_index.search(audio_embedding, top_k=top_k) if audio_embedding is not None else []

        visual_map = {hit["metadata"]["entry_id"]: hit for hit in visual_hits}
        audio_map = {hit["metadata"]["entry_id"]: hit for hit in audio_hits}
        all_ids = list(dict.fromkeys([*visual_map.keys(), *audio_map.keys()]))

        results: list[dict[str, Any]] = []
        for entry_id in all_ids:
            visual_hit = visual_map.get(entry_id)
            audio_hit = audio_map.get(entry_id)
            metadata = dict((visual_hit or audio_hit)["metadata"])
            visual_score = float(visual_hit["score"]) if visual_hit else None
            audio_score = float(audio_hit["score"]) if audio_hit else None

            if audio_score is None:
                fused = float(visual_score or 0.0)
            elif visual_score is None:
                fused = float(audio_score)
            else:
                fused = float(
                    (self.visual_weight * visual_score + self.audio_weight * audio_score)
                    / (self.visual_weight + self.audio_weight)
                )

            results.append(
                {
                    **metadata,
                    "visual_similarity": visual_score,
                    "audio_similarity": audio_score,
                    "fused_similarity": fused,
                }
            )

        return sorted(results, key=lambda item: item["fused_similarity"], reverse=True)[:top_k]

