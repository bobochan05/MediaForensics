from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID

import cv2
import numpy as np
from sqlalchemy.orm import Session

from ai.layer1_detection.frame_extractor import extract_sampled_frames
from ai.layer2_matching.similarity.embedding import VisualEmbeddingService
from ai.layer2_matching.similarity.faiss_index import FaissVectorIndex, normalize_rows
from ai.layer3_tracking.db import crud
from ai.layer3_tracking.db.database import SessionLocal, engine, ensure_layer3_schema
from ai.layer3_tracking.db.models import Content, ContentCluster, ContentStatus
from ai.layer3_tracking.services.alerting import AlertEvent, get_alert_service
from ai.layer3_tracking.services.risk_analyzer import RiskAnalyzer
from ai.layer3_tracking.services.url_utils import extract_domain, is_trusted_domain, normalize_urls
from ai.shared.video_budget import adaptive_frame_plan

LOGGER = logging.getLogger(__name__)

EMBEDDING_THRESHOLD = 0.86
RELATED_THRESHOLD = 0.76
EXACT_PHASH_DISTANCE = 5
NEAR_PHASH_DISTANCE = 10
RAPID_REAPPEARANCE_WINDOW = timedelta(hours=12)
LOW_RISK_TTL = timedelta(days=14)
RECENT_RESULT_TTL = timedelta(minutes=10)
RECENT_LOOKUP_TTL = timedelta(minutes=10)
RECENT_CACHE_LIMIT = 256
INDEX_SAVE_INTERVAL = 8
INDEX_SAVE_MAX_DELAY = timedelta(seconds=45)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_frame(path: Path):
    effective_sample_fps, effective_frames_per_video, _ = adaptive_frame_plan(
        path,
        purpose="layer3",
        requested_sample_fps=1.0,
        requested_frames_per_video=1,
    )
    frames = extract_sampled_frames(
        path,
        image_size=256,
        sample_fps=effective_sample_fps,
        frames_per_video=effective_frames_per_video,
        purpose="layer3",
    )
    if not frames:
        raise ValueError(f"No frame available for Layer 3 intelligence: {path}")
    return frames[0]


def _perceptual_hash(image) -> str:
    grayscale = np.asarray(image.convert("L").resize((32, 32)))
    reduced = grayscale.astype(np.float32)
    dct = cv2.dct(reduced)
    low_freq = dct[:8, :8]
    flattened = low_freq.flatten()
    median = float(np.median(flattened[1:])) if flattened.size > 1 else float(np.median(flattened))
    bits = (low_freq > median).astype(np.uint8).reshape(-1)
    return "".join("1" if int(bit) else "0" for bit in bits.tolist())


def _hamming_distance(left: str | None, right: str | None) -> int | None:
    if not left or not right or len(left) != len(right):
        return None
    return sum(ch1 != ch2 for ch1, ch2 in zip(left, right, strict=False))


def _media_type_for(path: Path) -> str:
    if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return "video"
    return "image"


def _save_embedding(path: Path, vector: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(vector, dtype=np.float16))
    return str(path)


def _load_embedding(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    return np.load(target).astype(np.float32)


def _extract_source_urls(layer2_payload: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in ("exact_matches", "visual_matches_top10", "related_web_sources", "matches"):
        items = layer2_payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                url = str(item.get("url") or item.get("source_url") or "").strip()
                if url:
                    urls.append(url)
    return normalize_urls(urls)


def _risk_level(score: float) -> str:
    if score >= 0.85:
        return "Critical"
    if score >= 0.65:
        return "High"
    if score >= 0.35:
        return "Medium"
    return "Low"


@dataclass(slots=True)
class Layer3IntelligenceResult:
    content_id: str
    cluster_id: str | None
    similar_count: int
    risk_score: float
    risk_level: str
    tracking_enabled: bool
    timeline_summary: dict[str, Any]
    timeline_points: list[dict[str, Any]]
    spread_indicators: dict[str, Any]
    propagation_graph: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_id": self.content_id,
            "cluster_id": self.cluster_id,
            "similar_count": self.similar_count,
            "risk_score": round(float(self.risk_score), 4),
            "risk_level": self.risk_level,
            "tracking_enabled": self.tracking_enabled,
            "timeline_summary": self.timeline_summary,
            "timeline_points": self.timeline_points,
            "spread_indicators": self.spread_indicators,
            "propagation_graph": self.propagation_graph,
        }


class Layer3IntelligenceStore:
    def __init__(self) -> None:
        ensure_layer3_schema(engine)
        self.root_dir = Path(__file__).resolve().parents[3] / "artifacts" / "layer3" / "intelligence"
        self.embedding_dir = self.root_dir / "embeddings"
        self.index_dir = self.root_dir / "indexes"
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = VisualEmbeddingService(device="cpu", sample_fps=0.35, max_frames_per_video=2, image_size=224)
        self.index = FaissVectorIndex("layer3_content_vectors", self.index_dir)
        self.risk_analyzer = RiskAnalyzer()
        self._lock = Lock()
        self._last_prune_at: datetime | None = None
        self._recent_result_cache: dict[str, tuple[datetime, dict[str, Any]]] = {}
        self._recent_cluster_cache: dict[str, tuple[datetime, tuple[str | None, int, float, int | None]]] = {}
        self._index_dirty_adds = 0
        self._last_index_save_at = _utcnow()

    def _prune_recent_caches(self, now: datetime) -> None:
        result_cutoff = now - RECENT_RESULT_TTL
        for key, (ts, _) in list(self._recent_result_cache.items()):
            if ts < result_cutoff:
                self._recent_result_cache.pop(key, None)
        lookup_cutoff = now - RECENT_LOOKUP_TTL
        for key, (ts, _) in list(self._recent_cluster_cache.items()):
            if ts < lookup_cutoff:
                self._recent_cluster_cache.pop(key, None)
        while len(self._recent_result_cache) > RECENT_CACHE_LIMIT:
            oldest_key = min(self._recent_result_cache.items(), key=lambda item: item[1][0])[0]
            self._recent_result_cache.pop(oldest_key, None)
        while len(self._recent_cluster_cache) > RECENT_CACHE_LIMIT:
            oldest_key = min(self._recent_cluster_cache.items(), key=lambda item: item[1][0])[0]
            self._recent_cluster_cache.pop(oldest_key, None)

    def _cache_result_key(self, *, content_hash: str, detection_score: float, source_urls: list[str], tracking_enabled: bool) -> str:
        return "|".join(
            [
                content_hash,
                f"{float(detection_score):.4f}",
                "1" if tracking_enabled else "0",
                ",".join(sorted(source_urls)[:12]),
            ]
        )

    def _cluster_lookup_key(self, *, content_hash: str, perceptual_hash: str) -> str:
        return f"{content_hash}|{perceptual_hash}"

    def _maybe_save_index(self, *, force: bool = False) -> None:
        now = _utcnow()
        if not force:
            if self._index_dirty_adds <= 0:
                return
            if self._index_dirty_adds < INDEX_SAVE_INTERVAL and (now - self._last_index_save_at) < INDEX_SAVE_MAX_DELAY:
                return
        self.index.save()
        self._index_dirty_adds = 0
        self._last_index_save_at = now

    def _find_cluster_match(
        self,
        session: Session,
        *,
        content_hash: str,
        perceptual_hash: str,
        embedding: np.ndarray,
    ) -> tuple[ContentCluster | None, int, float, int | None]:
        cache_key = self._cluster_lookup_key(content_hash=content_hash, perceptual_hash=perceptual_hash)
        cached = self._recent_cluster_cache.get(cache_key)
        if cached and (_utcnow() - cached[0]) <= RECENT_LOOKUP_TTL:
            cluster_id, similar_count, best_score, best_distance = cached[1]
            cluster = crud.get_cluster(session, UUID(cluster_id)) if cluster_id else None
            if cluster_id is None or cluster is not None:
                return cluster, similar_count, best_score, best_distance

        similar_count = 0
        best_cluster: ContentCluster | None = None
        best_score = 0.0
        best_distance: int | None = None
        for result in self.index.search(embedding, top_k=12):
            metadata = dict(result.get("metadata") or {})
            raw_content_id = str(metadata.get("content_id") or "").strip()
            if not raw_content_id:
                continue
            try:
                content_id = UUID(raw_content_id)
            except ValueError:
                continue
            if str(metadata.get("content_hash") or "") == content_hash:
                continue
            candidate = crud.get_content(session, content_id)
            if candidate is None:
                continue
            score = float(result.get("score") or 0.0)
            distance = _hamming_distance(perceptual_hash, candidate.perceptual_hash)
            qualifies = (
                (distance is not None and distance <= NEAR_PHASH_DISTANCE)
                or score >= RELATED_THRESHOLD
            )
            if not qualifies:
                continue
            similar_count += 1
            if candidate.cluster is None:
                continue
            priority = (
                distance is not None and distance <= NEAR_PHASH_DISTANCE,
                score,
                -(distance or 99),
            )
            best_priority = (
                best_distance is not None and best_distance <= NEAR_PHASH_DISTANCE,
                best_score,
                -(best_distance or 99),
            )
            if best_cluster is None or priority > best_priority:
                best_cluster = candidate.cluster
                best_score = score
                best_distance = distance
        self._recent_cluster_cache[cache_key] = (
            _utcnow(),
            (str(best_cluster.id) if best_cluster is not None else None, similar_count, best_score, best_distance),
        )
        return best_cluster, similar_count, best_score, best_distance

    def _rebuild_index(self, session: Session) -> None:
        rebuilt = FaissVectorIndex("layer3_content_vectors", self.index_dir)
        rebuilt.index = None
        rebuilt.dimension = None
        rebuilt.metadata = []
        vectors: list[np.ndarray] = []
        metadata: list[dict[str, object]] = []
        for cluster in crud.list_clusters(session):
            for content in cluster.contents:
                vector = _load_embedding(content.embedding_path)
                if vector is None:
                    continue
                vectors.append(vector.astype(np.float32))
                metadata.append(
                    {
                        "content_id": str(content.id),
                        "cluster_id": str(cluster.id),
                        "content_hash": content.hash,
                        "perceptual_hash": content.perceptual_hash,
                        "media_type": content.media_type,
                    }
                )
        if vectors:
            rebuilt.add(np.asarray(vectors, dtype=np.float32), metadata)
        rebuilt.save()
        self.index = rebuilt
        self._index_dirty_adds = 0
        self._last_index_save_at = _utcnow()

    def _prune_stale_content(self, session: Session, checked_at: datetime) -> None:
        if self._last_prune_at and (checked_at - self._last_prune_at) < timedelta(hours=6):
            return
        stale_before = checked_at - LOW_RISK_TTL
        removed = False
        for cluster in crud.list_clusters(session):
            active_contents: list[Content] = []
            for content in list(cluster.contents):
                stale = bool(
                    content.last_checked
                    and content.last_checked < stale_before
                    and float(content.risk_score or 0.0) < 0.25
                    and not content.tracking_enabled
                )
                if stale:
                    if content.embedding_path:
                        Path(content.embedding_path).unlink(missing_ok=True)
                    session.delete(content)
                    removed = True
                else:
                    active_contents.append(content)
            if not active_contents:
                if cluster.centroid_path:
                    Path(cluster.centroid_path).unlink(missing_ok=True)
                session.delete(cluster)
                removed = True
                continue
            cluster.content_count = len(active_contents)
            cluster.last_seen = max((item.last_checked or checked_at) for item in active_contents)
        if removed:
            session.flush()
            self._rebuild_index(session)
        self._last_prune_at = checked_at

    def _store_sources_and_tracking(
        self,
        session: Session,
        *,
        content: Content,
        source_urls: list[str],
        checked_at: datetime,
    ) -> tuple[float, ContentStatus, dict[str, Any], dict[str, Any]]:
        existing_urls = crud.get_source_urls(session, content.id)
        new_urls = [url for url in source_urls if url not in existing_urls]
        reused_urls = [url for url in source_urls if url in existing_urls]
        crud.touch_existing_sources(session, content.id, reused_urls, checked_at)
        crud.insert_new_sources(session, content.id, new_urls, checked_at)
        total_sources = crud.get_total_sources(session, content.id)
        previous_log = crud.get_previous_tracking_log(session, content.id, success_only=True)
        domains = [extract_domain(url) for url in source_urls if extract_domain(url)]
        unique_domains = sorted(set(domains))
        trusted_domains = sum(1 for domain in unique_domains if is_trusted_domain(domain))
        analysis = self.risk_analyzer.analyze(
            previous_total=len(existing_urls),
            new_sources=len(new_urls),
            total_sources=total_sources,
            checked_at=checked_at,
            previous_checked_at=previous_log.checked_at if previous_log else None,
            trusted_domains=trusted_domains,
            unknown_domains=max(0, len(unique_domains) - trusted_domains),
        )
        crud.create_tracking_log(
            session,
            content_id=content.id,
            checked_at=checked_at,
            total_sources=total_sources,
            new_sources=len(new_urls),
            growth_rate=analysis.growth_rate,
            growth_velocity=analysis.growth_velocity,
            spread_score=analysis.spread_score,
            risk_score=analysis.risk_score,
        )
        timeline_summary = {
            "first_seen": checked_at.isoformat(),
            "last_seen": checked_at.isoformat(),
            "observations": total_sources,
            "new_sources": len(new_urls),
            "total_sources": total_sources,
        }
        spread_indicators = {
            "spread_velocity": round(float(analysis.growth_velocity), 4),
            "platform_diversity": len(unique_domains),
            "domains": unique_domains[:8],
        }
        return analysis.risk_score, analysis.status, timeline_summary, spread_indicators

    def _update_cluster(
        self,
        session: Session,
        *,
        cluster: ContentCluster | None,
        embedding: np.ndarray,
        embedding_path: str,
        perceptual_hash: str,
        checked_at: datetime,
        risk_score: float,
        tracking_enabled: bool,
        is_new_content: bool,
    ) -> ContentCluster:
        if cluster is None:
            return crud.create_cluster(
                session,
                centroid_path=embedding_path,
                centroid_hash=perceptual_hash,
                content_count=1,
                risk_score=risk_score,
                tracking_enabled=tracking_enabled,
            )

        content_count = int(cluster.content_count or 0)
        previous_centroid = _load_embedding(cluster.centroid_path) or embedding
        divisor = max(content_count, 1)
        updated_centroid = normalize_rows(((previous_centroid * divisor) + embedding).reshape(1, -1))[0]
        centroid_path = _save_embedding(self.embedding_dir / f"{cluster.id}_centroid.npy", updated_centroid)
        next_count = content_count + 1 if is_new_content else max(content_count, 1)
        return crud.update_cluster(
            session,
            cluster=cluster,
            centroid_path=centroid_path,
            centroid_hash=perceptual_hash,
            content_count=next_count,
            risk_score=max(float(cluster.risk_score or 0.0), risk_score),
            tracking_enabled=bool(cluster.tracking_enabled or tracking_enabled),
            last_seen=checked_at,
        )

    def persist_analysis(
        self,
        *,
        media_path: str | Path,
        detection_score: float,
        layer2_payload: dict[str, Any],
        media_url: str | None = None,
        track_requested: bool = False,
        allow_alerting: bool = False,
        owner_user_id: int | None = None,
        session_scope_id: str | None = None,
        alert_frequency: str | None = "immediate",
    ) -> dict[str, Any]:
        target = Path(media_path)
        checked_at = _utcnow()
        self._prune_recent_caches(checked_at)
        content_hash = _sha256_file(target)
        first_frame = _first_frame(target)
        perceptual_hash = _perceptual_hash(first_frame)
        media_type = _media_type_for(target)
        source_urls = _extract_source_urls(layer2_payload)
        result_cache_key = self._cache_result_key(
            content_hash=content_hash,
            detection_score=float(detection_score),
            source_urls=source_urls,
            tracking_enabled=bool(track_requested),
        )

        with self._lock:
            with SessionLocal.begin() as session:
                self._prune_stale_content(session, checked_at)
                existing = crud.get_content_by_hash(session, content_hash, for_update=True)
                if existing is None and perceptual_hash:
                    existing = crud.get_content_by_perceptual_hash(session, perceptual_hash, for_update=True)
                if existing is not None:
                    cached_result = self._recent_result_cache.get(result_cache_key)
                    if cached_result and (checked_at - cached_result[0]) <= RECENT_RESULT_TTL:
                        self._store_sources_and_tracking(
                            session,
                            content=existing,
                            source_urls=source_urls,
                            checked_at=checked_at,
                        )
                        return dict(cached_result[1])

                embedding: np.ndarray | None = None
                embedding_path = str(self.embedding_dir / f"{content_hash}.npy")
                cluster = existing.cluster if existing is not None else None
                similar_count = int(existing.similar_count or 0) if existing is not None else 0
                best_score = 1.0 if existing is not None else 0.0
                best_distance = 0 if existing is not None else None
                if existing is None:
                    embedding = self.embedder.embed_media(target)
                    embedding_path = _save_embedding(self.embedding_dir / f"{content_hash}.npy", embedding)
                    cluster, similar_count, best_score, best_distance = self._find_cluster_match(
                        session,
                        content_hash=content_hash,
                        perceptual_hash=perceptual_hash,
                        embedding=embedding,
                    )
                rapid_reappearance = bool(existing and existing.last_checked and (checked_at - existing.last_checked) <= RAPID_REAPPEARANCE_WINDOW)

                if existing is None:
                    provisional_tracking = bool(track_requested or detection_score >= 0.8 or similar_count >= 2)
                    cluster = self._update_cluster(
                        session,
                        cluster=cluster,
                        embedding=embedding if embedding is not None else _load_embedding(embedding_path) or np.zeros((512,), dtype=np.float32),
                        embedding_path=embedding_path,
                        perceptual_hash=perceptual_hash,
                        checked_at=checked_at,
                        risk_score=float(detection_score),
                        tracking_enabled=provisional_tracking,
                        is_new_content=True,
                    )
                    content = crud.create_content(
                        session,
                        content_hash=content_hash,
                        perceptual_hash=perceptual_hash,
                        media_type=media_type,
                        detection_score=float(detection_score),
                        embedding_path=embedding_path,
                        media_url=media_url,
                        owner_user_id=owner_user_id,
                        session_scope_id=session_scope_id,
                        risk_score=float(detection_score),
                        cluster_id=cluster.id,
                        similar_count=similar_count,
                        tracking_enabled=provisional_tracking,
                        alert_email_enabled=bool(allow_alerting and provisional_tracking),
                        alert_frequency=alert_frequency if allow_alerting else None,
                    )
                    self.index.add(
                        np.asarray([embedding], dtype=np.float32),
                        [
                            {
                                "content_id": str(content.id),
                                "cluster_id": str(cluster.id),
                                "content_hash": content_hash,
                                "perceptual_hash": perceptual_hash,
                                "media_type": media_type,
                            }
                        ],
                    )
                    self._index_dirty_adds += 1
                    self._maybe_save_index()
                else:
                    content = existing
                    cluster = content.cluster or cluster
                    if cluster is None:
                        seed_embedding = _load_embedding(content.embedding_path) or self.embedder.embed_media(target)
                        cluster = self._update_cluster(
                            session,
                            cluster=None,
                            embedding=seed_embedding,
                            embedding_path=content.embedding_path or embedding_path,
                            perceptual_hash=perceptual_hash,
                            checked_at=checked_at,
                            risk_score=float(detection_score),
                            tracking_enabled=False,
                            is_new_content=False,
                        )

                base_risk_score, status, timeline_summary, spread_indicators = self._store_sources_and_tracking(
                    session,
                    content=content,
                    source_urls=source_urls,
                    checked_at=checked_at,
                )
                cluster_size = int(cluster.content_count or len(cluster.contents or []))
                exact_or_near = best_distance is not None and best_distance <= NEAR_PHASH_DISTANCE
                mutation_signal = int((layer2_payload.get("counts") or {}).get("visual") or 0) + int((layer2_payload.get("counts") or {}).get("embedding") or 0)
                mutation_rate = 0.0 if cluster_size <= 1 else round(min(1.0, mutation_signal / max(cluster_size, 1)), 4)
                final_tracking = bool(
                    track_requested
                    or detection_score >= 0.8
                    or cluster_size >= 3
                    or rapid_reappearance
                    or len(source_urls) >= 5
                )
                final_risk = min(
                    1.0,
                    max(
                        base_risk_score,
                        float(detection_score) * 0.5,
                        min(1.0, cluster_size / 10.0) * 0.25 + min(1.0, len(source_urls) / 8.0) * 0.25,
                    ),
                )
                spread_indicators.update(
                    {
                        "mutation_rate": mutation_rate,
                        "cluster_size": cluster_size,
                        "rapid_reappearance": rapid_reappearance,
                        "strong_match_similarity": round(float(best_score), 4),
                        "hash_match": exact_or_near,
                    }
                )
                tracking_history = crud.get_tracking_history(session, content.id)
                if tracking_history:
                    timeline_summary["first_seen"] = tracking_history[0].checked_at.isoformat()
                    timeline_summary["last_seen"] = tracking_history[-1].checked_at.isoformat()
                    timeline_summary["observations"] = len(tracking_history)
                timeline_points = [
                    {
                        "timestamp": log.checked_at.isoformat(),
                        "mentions": int(log.total_sources),
                    }
                    for log in tracking_history[-24:]
                ]
                crud.update_content_intelligence(
                    session,
                    content=content,
                    perceptual_hash=perceptual_hash,
                    media_type=media_type,
                    detection_score=float(detection_score),
                    embedding_path=embedding_path,
                    media_url=media_url,
                    owner_user_id=owner_user_id,
                    session_scope_id=session_scope_id,
                    cluster_id=cluster.id if cluster else None,
                    similar_count=similar_count,
                    tracking_enabled=final_tracking,
                    alert_email_enabled=bool(allow_alerting and final_tracking),
                    alert_frequency=alert_frequency if allow_alerting else None,
                    risk_score=final_risk,
                    status=status,
                    last_checked=checked_at,
                )
                cluster = self._update_cluster(
                    session,
                    cluster=cluster,
                    embedding=_load_embedding(content.embedding_path) or embedding or self.embedder.embed_media(target),
                    embedding_path=embedding_path,
                    perceptual_hash=perceptual_hash,
                    checked_at=checked_at,
                    risk_score=final_risk,
                    tracking_enabled=final_tracking,
                    is_new_content=False,
                )
                result = Layer3IntelligenceResult(
                    content_id=str(content.id),
                    cluster_id=str(cluster.id) if cluster else None,
                    similar_count=max(similar_count, max(cluster_size - 1, 0)),
                    risk_score=final_risk,
                    risk_level=_risk_level(final_risk),
                    tracking_enabled=final_tracking,
                    timeline_summary=timeline_summary,
                    timeline_points=timeline_points,
                    spread_indicators=spread_indicators,
                    propagation_graph={
                        "nodes": max(cluster_size, 1),
                        "edges": max(cluster_size - 1, 0),
                        "variation_detected": bool(mutation_rate > 0.0 or exact_or_near),
                    },
                )
                LOGGER.info(
                    "Layer3 intelligence stored content_id=%s cluster_id=%s similar=%s risk=%.4f tracking=%s",
                    result.content_id,
                    result.cluster_id,
                    result.similar_count,
                    result.risk_score,
                    result.tracking_enabled,
                )
                if allow_alerting and result.tracking_enabled and result.risk_score > 0.85:
                    get_alert_service().trigger_alert(
                        AlertEvent(
                            event_type="high_risk_content_detected",
                            severity="HIGH",
                            message="Layer 3 flagged this content cluster as high risk.",
                            content_id=result.content_id,
                            cluster_id=result.cluster_id,
                            explanation=f"risk_score={result.risk_score:.4f}, similar_count={result.similar_count}",
                            metadata={"risk_score": result.risk_score, "risk_level": result.risk_level},
                        )
                    )
                if allow_alerting and result.tracking_enabled and rapid_reappearance:
                    get_alert_service().trigger_alert(
                        AlertEvent(
                            event_type="rapid_reappearance_detected",
                            severity="HIGH",
                            message="Content reappeared again within the short monitoring window.",
                            content_id=result.content_id,
                            cluster_id=result.cluster_id,
                            explanation=f"rapid_window_hours={RAPID_REAPPEARANCE_WINDOW.total_seconds() / 3600:.1f}",
                            metadata={"spread_velocity": spread_indicators.get("spread_velocity", 0.0)},
                        )
                    )
                if allow_alerting and result.tracking_enabled and float(spread_indicators.get("spread_velocity") or 0.0) > 0.5:
                    get_alert_service().trigger_alert(
                        AlertEvent(
                            event_type="spread_velocity_spike",
                            severity="HIGH",
                            message="Spread velocity exceeded the expected threshold.",
                            content_id=result.content_id,
                            cluster_id=result.cluster_id,
                            explanation=f"spread_velocity={float(spread_indicators.get('spread_velocity') or 0.0):.4f}",
                            metadata={"similar_count": result.similar_count},
                        )
                    )
                if allow_alerting and result.tracking_enabled and cluster_size >= 10:
                    get_alert_service().trigger_alert(
                        AlertEvent(
                            event_type="cluster_explosion",
                            severity="CRITICAL" if cluster_size >= 20 else "HIGH",
                            message="Cluster size increased sharply and may indicate coordinated redistribution.",
                            content_id=result.content_id,
                            cluster_id=result.cluster_id,
                            explanation=f"cluster_size={cluster_size}",
                            metadata={"cluster_size": cluster_size, "mutation_rate": mutation_rate},
                        )
                    )
                payload = result.to_dict()
                self._recent_result_cache[result_cache_key] = (checked_at, dict(payload))
                return payload
