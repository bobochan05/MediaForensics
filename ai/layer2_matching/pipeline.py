from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ai.layer2_matching.audio.audio_embedding import AudioEmbeddingResult, AudioEmbeddingService
from ai.layer2_matching.audio.audio_extract import VIDEO_EXTENSIONS, extract_audio_from_media
from ai.layer1_detection.data_loader import collect_faceforensics_videos
from ai.layer1_detection.inference import predict_video
from ai.layer2_matching.insights import build_layer2_insights
from ai.layer2_matching.schemas import AnalysisResponse, OriginEstimate, RiskBreakdown, SimilarContentItem, TimelinePoint
from ai.layer2_matching.nlp.context_classifier import ContextClassifier
from ai.layer2_matching.risk.scoring import compute_risk_assessment
from ai.layer2_matching.similarity.embedding import VisualEmbeddingService
from ai.layer2_matching.similarity.faiss_index import FaissVectorIndex
from ai.layer2_matching.similarity.search import MultimodalSimilaritySearch
from ai.layer2_matching.tracking.external_search import ExternalSearchClient
from ai.layer2_matching.tracking.metadata_parser import OccurrenceRecord, credibility_score_for_source, normalize_timestamp, timestamp_from_path
from ai.layer2_matching.tracking.timeline import build_spread_timeline, estimate_origin
from ai.shared.file_utils import ensure_dir, load_json, load_numpy, save_json


LOGGER = logging.getLogger(__name__)


@dataclass
class Layer2Config:
    project_dir: Path
    artifacts_dir: Path
    layer2_dir: Path
    uploads_dir: Path
    analyses_dir: Path
    cache_dir: Path
    index_dir: Path
    sample_fps: float = 0.35
    max_frames_per_video: int = 4
    max_seed_items: int = 512
    max_audio_seed_items: int = 32
    top_k: int = 10
    device: str = "auto"


class Layer2Pipeline:
    def __init__(self, project_dir: str | Path) -> None:
        project_dir = Path(project_dir).resolve()
        artifacts_dir = project_dir / "artifacts"
        layer2_dir = ensure_dir(artifacts_dir / "layer2")
        self.config = Layer2Config(
            project_dir=project_dir,
            artifacts_dir=artifacts_dir,
            layer2_dir=layer2_dir,
            uploads_dir=ensure_dir(layer2_dir / "uploads"),
            analyses_dir=ensure_dir(layer2_dir / "analyses"),
            cache_dir=ensure_dir(layer2_dir / "cache"),
            index_dir=ensure_dir(layer2_dir / "indexes"),
        )
        self.visual_embedder = VisualEmbeddingService(
            device=self.config.device,
            sample_fps=self.config.sample_fps,
            max_frames_per_video=self.config.max_frames_per_video,
        )
        self.audio_embedder = AudioEmbeddingService(device=self.config.device)
        self.context_classifier = ContextClassifier(device=self.config.device)
        self.external_search = ExternalSearchClient(
            cache_dir=self.config.cache_dir / "reverse_search",
            visual_embedder=self.visual_embedder,
            audio_verifier_embedder=self.audio_embedder,
            sample_fps=self.config.sample_fps,
            max_frames_per_video=self.config.max_frames_per_video,
        )
        self.visual_index = FaissVectorIndex("visual", self.config.index_dir)
        self.audio_index = FaissVectorIndex("audio", self.config.index_dir)
        self.search_engine = MultimodalSimilaritySearch(self.visual_index, self.audio_index)

    def _analysis_path(self, analysis_id: str) -> Path:
        return self.config.analyses_dir / f"{analysis_id}.json"

    @staticmethod
    def _entry_id(seed: str) -> str:
        return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]

    def _resolve_local_path(self, path: str | Path) -> Path:
        path_obj = Path(path)
        candidates: list[Path] = []

        if path_obj.exists():
            return path_obj.resolve()

        if path_obj.is_absolute():
            candidates.append(path_obj)
        else:
            candidates.append(self.config.project_dir / path_obj)

        parts = list(path_obj.parts)
        lower_parts = [part.lower() for part in parts]
        if lower_parts:
            if lower_parts[0] == "dataset":
                relative = Path(*parts[1:]) if len(parts) > 1 else Path()
                candidates.append(self.config.project_dir / "data" / "datasets" / relative)
            if len(lower_parts) >= 2 and lower_parts[0] == "data" and lower_parts[1] == "dataset":
                relative = Path(*parts[2:]) if len(parts) > 2 else Path()
                candidates.append(self.config.project_dir / "data" / "datasets" / relative)
            if len(lower_parts) >= 2 and lower_parts[0] == "data" and lower_parts[1] == "datasets":
                relative = Path(*parts[2:]) if len(parts) > 2 else Path()
                candidates.append(self.config.project_dir / "data" / "datasets" / relative)
            artifact_markers = {
                ("artifacts", "layer2", "uploads"): self.config.project_dir / "artifacts" / "layer2" / "uploads",
                ("artifacts", "ui_uploads"): self.config.project_dir / "artifacts" / "ui_uploads",
            }
            for marker_parts, marker_target in artifact_markers.items():
                marker_length = len(marker_parts)
                for index in range(0, len(lower_parts) - marker_length + 1):
                    if tuple(lower_parts[index : index + marker_length]) != marker_parts:
                        continue
                    remainder = Path(*parts[index + marker_length :]) if len(parts) > index + marker_length else Path()
                    candidates.append(marker_target / remainder)
                    break

        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                return candidate.resolve()

        return candidates[0] if candidates else path_obj

    def _store_upload(self, source_path: Path, analysis_id: str, original_filename: str | None) -> Path:
        filename = Path(original_filename or source_path.name).name
        target_path = self.config.uploads_dir / f"{analysis_id}_{filename}"
        shutil.copy2(source_path, target_path)
        return target_path

    def _infer_layer1(self, media_path: Path) -> tuple[bool, float]:
        label, confidence = predict_video(
            video_path=media_path,
            classifier_path=self.config.artifacts_dir / "fusion_model.pth",
            efficientnet_path=self.config.artifacts_dir / "efficientnet_finetuned.pth",
            dino_path=self.config.artifacts_dir / "dino_finetuned.pth",
            sample_fps=self.config.sample_fps,
            device=self.config.device,
        )
        return label == "fake", float(confidence)

    def _local_metadata(self, path: str | Path, label: str | None, source: str = "layer1_cache") -> dict[str, object]:
        path_obj = self._resolve_local_path(path)
        return {
            "entry_id": self._entry_id(str(path_obj.resolve()) if path_obj.exists() else str(path_obj)),
            "source_type": "local",
            "platform": "local_dataset",
            "local_path": str(path_obj),
            "url": None,
            "timestamp": timestamp_from_path(path_obj),
            "title": path_obj.name,
            "caption": f"Local corpus item sourced from {source}.",
            "label": label,
            "credibility_score": credibility_score_for_source("local_dataset"),
            "context": "news",
            "context_scores": {"news": 0.4, "meme": 0.2, "propaganda / misinformation": 0.4},
            "metadata": {"origin": source},
        }

    def _seed_from_layer1_embeddings(self) -> tuple[np.ndarray, list[dict[str, object]]]:
        grouped: dict[str, dict[str, object]] = {}
        for split_name in ("train", "test"):
            split_dir = self.config.artifacts_dir / "embeddings" / split_name
            clip_path = split_dir / "clip_embeddings.npy"
            paths_path = split_dir / "frame_video_paths.json"
            labels_path = split_dir / "labels.npy"
            if not clip_path.exists() or not paths_path.exists() or not labels_path.exists():
                continue

            clip_embeddings = load_numpy(clip_path)
            frame_paths = list(load_json(paths_path))
            labels = load_numpy(labels_path)
            for embedding, file_path, label in zip(clip_embeddings, frame_paths, labels, strict=False):
                key = str(file_path)
                bucket = grouped.setdefault(key, {"vectors": [], "label": "fake" if int(label) == 1 else "real"})
                bucket["vectors"].append(np.asarray(embedding, dtype=np.float32))

        if not grouped:
            return np.empty((0, 512), dtype=np.float32), []

        sorted_paths = sorted(grouped)
        if len(sorted_paths) > self.config.max_seed_items:
            indices = np.linspace(0, len(sorted_paths) - 1, num=self.config.max_seed_items, dtype=int)
            sorted_paths = [sorted_paths[index] for index in indices]

        vectors: list[np.ndarray] = []
        metadata: list[dict[str, object]] = []
        for file_path in sorted_paths:
            bucket = grouped[file_path]
            vectors.append(np.mean(np.stack(bucket["vectors"], axis=0), axis=0).astype(np.float32))
            metadata.append(self._local_metadata(file_path, str(bucket["label"])))
        return np.vstack(vectors).astype(np.float32), metadata

    def _seed_from_dataset_scan(self) -> tuple[np.ndarray, list[dict[str, object]]]:
        dataset_dir = self.config.project_dir / "data" / "datasets"
        if not dataset_dir.exists():
            return np.empty((0, 512), dtype=np.float32), []

        video_records = collect_faceforensics_videos(dataset_dir=dataset_dir)
        selected_records = video_records[: self.config.max_seed_items]
        vectors: list[np.ndarray] = []
        metadata: list[dict[str, object]] = []
        for record in selected_records:
            try:
                embedding = self.visual_embedder.embed_media(record.video_path)
            except Exception:
                continue
            vectors.append(embedding.astype(np.float32))
            metadata.append(self._local_metadata(record.video_path, "fake" if int(record.label) == 1 else "real", source="dataset_scan"))

        if not vectors:
            return np.empty((0, 512), dtype=np.float32), []
        return np.vstack(vectors).astype(np.float32), metadata

    def _ensure_visual_index(self) -> None:
        if self.visual_index.size > 0:
            return
        vectors, metadata = self._seed_from_layer1_embeddings()
        if len(vectors) == 0:
            vectors, metadata = self._seed_from_dataset_scan()
        if len(vectors) > 0:
            self.visual_index.add(vectors, metadata)
            self.visual_index.save()

    def _ensure_audio_index(self) -> None:
        if self.audio_index.size > 0:
            return

        seeded = 0
        for item in self.visual_index.metadata:
            local_path = item.get("local_path")
            if not local_path:
                continue
            path_obj = self._resolve_local_path(str(local_path))
            if not path_obj.exists():
                continue
            if path_obj.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            extracted = extract_audio_from_media(path_obj, self.config.cache_dir / "seed_audio")
            if not extracted.has_audio:
                continue
            try:
                audio_embedding = self.audio_embedder.embed_audio(
                    waveform=extracted.waveform,
                    sample_rate=extracted.sample_rate,
                    duration_seconds=extracted.duration_seconds,
                )
            except Exception:
                continue
            if audio_embedding.combined_embedding is None:
                continue

            self.audio_index.add(audio_embedding.combined_embedding.reshape(1, -1), [{**item, "entry_id": item["entry_id"]}])
            seeded += 1
            if seeded >= self.config.max_audio_seed_items:
                break

        if seeded > 0:
            self.audio_index.save()

    def ensure_indexes(self) -> None:
        self._ensure_visual_index()
        self._ensure_audio_index()

    def _enrich_context(self, occurrences: list[OccurrenceRecord]) -> list[OccurrenceRecord]:
        for occurrence in occurrences:
            text = " ".join(part for part in [occurrence.title or "", occurrence.caption or "", occurrence.platform] if part).strip()
            label, scores = self.context_classifier.classify(text)
            occurrence.context = label
            occurrence.context_scores = scores
        return occurrences

    def _to_occurrence_records(self, matches: list[dict[str, object]]) -> list[OccurrenceRecord]:
        records: list[OccurrenceRecord] = []
        for match in matches:
            local_path = match.get("local_path")
            resolved_local_path = None
            if local_path:
                candidate = self._resolve_local_path(str(local_path))
                resolved_local_path = str(candidate) if candidate.exists() else str(candidate)
            records.append(
                OccurrenceRecord(
                    entry_id=str(match["entry_id"]),
                    source_type=str(match.get("source_type", "local")),
                    platform=str(match.get("platform", "unknown")),
                    url=match.get("url"),
                    local_path=resolved_local_path,
                    timestamp=normalize_timestamp(match.get("timestamp")),
                    title=match.get("title"),
                    caption=match.get("caption"),
                    label=match.get("label"),
                    credibility_score=float(match.get("credibility_score", 0.5)),
                    visual_similarity=float(match["visual_similarity"]) if match.get("visual_similarity") is not None else None,
                    audio_similarity=float(match["audio_similarity"]) if match.get("audio_similarity") is not None else None,
                    fused_similarity=float(match.get("fused_similarity", 0.0)),
                    context=str(match.get("context", "news")),
                    context_scores=dict(match.get("context_scores", {})),
                    is_mock=bool(match.get("is_mock", False)),
                    metadata=dict(match.get("metadata", {})),
                )
            )
        return records

    def _make_upload_record(self, stored_path: Path, timestamp: str, label: str, query_hint: str | None) -> dict[str, object]:
        return {
            "entry_id": self._entry_id(str(stored_path.resolve())),
            "source_type": "upload",
            "platform": "local_upload",
            "local_path": str(stored_path),
            "url": None,
            "timestamp": timestamp,
            "title": stored_path.name,
            "caption": query_hint,
            "label": label,
            "credibility_score": credibility_score_for_source("local_upload"),
            "context": "news",
            "context_scores": {"news": 0.34, "meme": 0.33, "propaganda / misinformation": 0.33},
            "metadata": {"origin": "upload"},
        }

    def _persist_upload_to_index(
        self,
        upload_metadata: dict[str, object],
        visual_embedding: np.ndarray,
        audio_embedding: AudioEmbeddingResult,
    ) -> None:
        existing_visual_ids = {str(item.get("entry_id")) for item in self.visual_index.metadata}
        if str(upload_metadata["entry_id"]) not in existing_visual_ids:
            self.visual_index.add(visual_embedding.reshape(1, -1), [upload_metadata])
            self.visual_index.save()

        if audio_embedding.combined_embedding is None:
            return
        existing_audio_ids = {str(item.get("entry_id")) for item in self.audio_index.metadata}
        if str(upload_metadata["entry_id"]) not in existing_audio_ids:
            self.audio_index.add(audio_embedding.combined_embedding.reshape(1, -1), [upload_metadata])
            self.audio_index.save()

    def analyze_media(
        self,
        source_path: str | Path,
        original_filename: str | None = None,
        query_hint: str | None = None,
        source_url: str | None = None,
        is_fake: bool | None = None,
        confidence: float | None = None,
        internet_only: bool = False,
        allow_mock_fallback: bool = False,
    ) -> AnalysisResponse:
        source_path = Path(source_path)
        created_at = datetime.now(timezone.utc).isoformat()
        analysis_id = self._entry_id(f"{source_path.name}|{created_at}")
        stored_path = self._store_upload(source_path, analysis_id, original_filename)

        if is_fake is None or confidence is None:
            is_fake, confidence = self._infer_layer1(stored_path)
        confidence = float(confidence)
        fake_probability = confidence if is_fake else 1.0 - confidence

        visual_embedding = self.visual_embedder.embed_media(stored_path)
        audio_extraction = extract_audio_from_media(stored_path, self.config.cache_dir / "uploads")
        audio_embedding = self.audio_embedder.embed_audio(
            waveform=audio_extraction.waveform,
            sample_rate=audio_extraction.sample_rate,
            duration_seconds=audio_extraction.duration_seconds,
        )

        local_matches: list[dict[str, object]] = []
        local_records: list[OccurrenceRecord] = []
        if not internet_only:
            self.ensure_indexes()
            local_matches = self.search_engine.search(
                visual_embedding=visual_embedding,
                audio_embedding=audio_embedding.combined_embedding,
                top_k=self.config.top_k,
            )
            local_records = self._to_occurrence_records(local_matches)
        external_records = self.external_search.search(
            file_path=stored_path,
            query_hint=query_hint,
            local_matches=local_matches,
            max_results=self.config.top_k,
            allow_mock_fallback=allow_mock_fallback,
            original_visual_embedding=visual_embedding,
            original_audio_embedding=audio_embedding.combined_embedding,
            source_url=source_url,
        )

        all_records = self._enrich_context(external_records if internet_only else [*local_records, *external_records])
        timeline = build_spread_timeline(all_records)
        origin = estimate_origin(all_records)
        risk = compute_risk_assessment(fake_probability=fake_probability, occurrences=all_records, timeline=timeline)

        similar_content = [
            SimilarContentItem(
                id=record.entry_id,
                url=record.url,
                media_url=str(record.metadata.get("media_url")) if record.metadata.get("media_url") else None,
                local_path=record.local_path,
                visual_similarity=record.visual_similarity,
                audio_similarity=record.audio_similarity,
                fused_similarity=record.fused_similarity,
                combined_score=record.fused_similarity,
                platform=record.platform,
                timestamp=record.timestamp,
                source_type=record.source_type,
                title=record.title,
                caption=record.caption,
                context=record.context,
                context_scores=record.context_scores,
                credibility_score=record.credibility_score,
                is_mock=record.is_mock,
                label=record.label,
                metadata={**record.metadata, **({"source_url": source_url} if source_url else {})},
            )
            for record in sorted(all_records, key=lambda item: item.fused_similarity, reverse=True)
        ]

        layer2_insights: dict[str, object] = {}
        try:
            layer2_insights = build_layer2_insights(
                similar_content=similar_content,
                original_media_path=stored_path,
                original_filename=original_filename or stored_path.name,
                query_hint=query_hint,
                audio_embedding_dim=int(audio_embedding.combined_embedding.shape[0]) if audio_embedding.combined_embedding is not None else None,
            )
        except Exception:
            LOGGER.exception("Layer 2 insights enrichment failed for %s", stored_path)
            layer2_insights = {}

        response = AnalysisResponse(
            analysis_id=analysis_id,
            filename=stored_path.name,
            media_type="video" if stored_path.suffix.lower() in VIDEO_EXTENSIONS else "image",
            is_fake=bool(is_fake),
            confidence=confidence,
            similar_content=similar_content,
            matches=similar_content,
            origin_estimate=OriginEstimate(**origin),
            spread_timeline=[TimelinePoint(**point) for point in timeline],
            risk_score=risk.score,
            risk_level=risk.level,
            risk_breakdown=RiskBreakdown(**risk.breakdown),
            confidence_explanation=risk.explanation,
            visual_embedding_dim=int(visual_embedding.shape[0]),
            audio_embedding_dim=int(audio_embedding.combined_embedding.shape[0]) if audio_embedding.combined_embedding is not None else None,
            external_matches_used=bool(external_records),
            created_at=created_at,
            layer2_insights=layer2_insights,
        )

        upload_metadata = self._make_upload_record(
            stored_path=stored_path,
            timestamp=created_at,
            label="fake" if is_fake else "real",
            query_hint=query_hint,
        )
        self._persist_upload_to_index(upload_metadata, visual_embedding, audio_embedding)
        save_json(self._analysis_path(analysis_id), response.model_dump(mode="json"))
        return response

    def find_local_similar(
        self,
        source_path: str | Path,
        source_url: str | None = None,
        top_k: int = 100,
    ) -> list[SimilarContentItem]:
        source_path = Path(source_path)
        self.ensure_indexes()

        visual_embedding = self.visual_embedder.embed_media(source_path)
        audio_extraction = extract_audio_from_media(source_path, self.config.cache_dir / "uploads")
        audio_embedding = self.audio_embedder.embed_audio(
            waveform=audio_extraction.waveform,
            sample_rate=audio_extraction.sample_rate,
            duration_seconds=audio_extraction.duration_seconds,
        )

        local_matches = self.search_engine.search(
            visual_embedding=visual_embedding,
            audio_embedding=audio_embedding.combined_embedding,
            top_k=int(top_k),
        )
        local_records = self._enrich_context(self._to_occurrence_records(local_matches))

        return [
            SimilarContentItem(
                id=record.entry_id,
                url=record.url,
                media_url=str(record.metadata.get("media_url")) if record.metadata.get("media_url") else None,
                local_path=record.local_path,
                visual_similarity=record.visual_similarity,
                audio_similarity=record.audio_similarity,
                fused_similarity=record.fused_similarity,
                combined_score=record.fused_similarity,
                platform=record.platform,
                timestamp=record.timestamp,
                source_type=record.source_type,
                title=record.title,
                caption=record.caption,
                context=record.context,
                context_scores=record.context_scores,
                credibility_score=record.credibility_score,
                is_mock=record.is_mock,
                label=record.label,
                metadata={**record.metadata, **({"source_url": source_url} if source_url else {})},
            )
            for record in sorted(local_records, key=lambda item: item.fused_similarity, reverse=True)
        ]

    def load_analysis(self, analysis_id: str) -> AnalysisResponse:
        analysis_path = self._analysis_path(analysis_id)
        if not analysis_path.exists():
            raise FileNotFoundError(f"No Layer 2 analysis found for id '{analysis_id}'.")
        return AnalysisResponse.model_validate(load_json(analysis_path))

    def load_similar(self, analysis_id: str) -> list[SimilarContentItem]:
        return self.load_analysis(analysis_id).similar_content
