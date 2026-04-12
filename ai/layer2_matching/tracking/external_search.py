from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse
from urllib.parse import urlparse

import numpy as np

from ai.layer1_detection.frame_extractor import IMAGE_EXTENSIONS, extract_sampled_frames
from ai.layer2_matching.similarity.embedding import VisualEmbeddingService
from ai.layer2_matching.similarity.verification import MultimodalVerificationService
from ai.layer2_matching.tracking.media_resolver import RemoteMediaResolver
from ai.layer2_matching.tracking.metadata_parser import OccurrenceRecord, credibility_score_for_source, infer_platform, normalize_timestamp
from ai.layer2_matching.tracking.query_fallback import QueryFallbackSearchClient
from ai.layer2_matching.tracking.reverse_search_service import upload_path_to_cloudinary
from ai.layer2_matching.tracking.reverse_image_providers import (
    ReverseImageCandidate,
    configured_reverse_image_providers,
    reverse_query_cache_key,
    reverse_search_provider_status,
)
from ai.shared.file_utils import ensure_dir, load_json, save_json


class ExternalSearchClient:
    """Primary internet discovery path based on reverse-search + embedding verification."""

    PROVIDER_CACHE_VERSION = "v3"

    def __init__(
        self,
        cache_dir: str | Path,
        visual_embedder: VisualEmbeddingService,
        audio_verifier_embedder,
        sample_fps: float = 0.5,
        max_frames_per_video: int = 8,
        verification_visual_threshold: float = 0.8,
        verification_audio_threshold: float = 0.75,
        timeout_seconds: int = 20,
    ) -> None:
        self.cache_dir = ensure_dir(cache_dir)
        self.sample_fps = float(sample_fps)
        self.max_frames_per_video = int(max_frames_per_video)
        self.timeout_seconds = int(timeout_seconds)
        self.visual_embedder = visual_embedder
        self.reverse_providers = configured_reverse_image_providers()
        self.reverse_provider_status = reverse_search_provider_status()
        self.query_fallback = QueryFallbackSearchClient(timeout_seconds=timeout_seconds)
        self.resolver = RemoteMediaResolver(self.cache_dir / "remote_media", timeout_seconds=timeout_seconds)
        self.verifier = MultimodalVerificationService(
            visual_embedder=visual_embedder,
            audio_embedder=audio_verifier_embedder,
            cache_dir=self.cache_dir / "verification",
            visual_threshold=verification_visual_threshold,
            audio_threshold=verification_audio_threshold,
        )
        self.query_frame_dir = ensure_dir(self.cache_dir / "reverse_query_frames")
        self.provider_cache_dir = ensure_dir(self.cache_dir / "reverse_provider_cache")
        self.public_query_url_dir = ensure_dir(self.cache_dir / "public_query_urls")
        self.low_signal_domains = {
            "aliexpress.com",
            "amazon.com",
            "ebay.com",
            "etsy.com",
            "flipkart.com",
            "pinterest.com",
            "temu.com",
            "walmart.com",
            "alibaba.com",
            "macys.com",
        }
        self.generic_domains = {
            "wikipedia.org",
            "simple.wikipedia.org",
            "britannica.com",
            "dictionary.com",
            "merriam-webster.com",
        }
        self.generic_keywords = [
            "what is",
            "definition",
            "meaning of",
            "fictional character",
            "overview",
            "introduction",
            "history of",
        ]

    def _prepare_query_images(self, file_path: Path) -> list[Path]:
        frames = extract_sampled_frames(
            video_path=file_path,
            image_size=512,
            sample_fps=self.sample_fps,
            frames_per_video=None,
        )
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            frames = frames[: self.max_frames_per_video]

        if not frames:
            return []

        media_hash = hashlib.sha1(str(file_path.resolve()).encode("utf-8")).hexdigest()[:16]
        frame_dir = ensure_dir(self.query_frame_dir / media_hash)
        query_paths: list[Path] = []
        for index, frame in enumerate(frames, start=1):
            frame_path = frame_dir / f"frame_{index:02d}.jpg"
            if not frame_path.exists():
                frame.save(frame_path, format="JPEG", quality=92)
            query_paths.append(frame_path)
        return query_paths

    @staticmethod
    def _public_source_url(source_url: str | None) -> str | None:
        if not source_url:
            return None
        cleaned = str(source_url).strip()
        if cleaned.startswith("http://") or cleaned.startswith("https://"):
            return cleaned
        return None

    def _provider_cache_path(self, provider_name: str, image_path: Path) -> Path:
        versioned_provider = f"{provider_name}_{self.PROVIDER_CACHE_VERSION}"
        return self.provider_cache_dir / f"{reverse_query_cache_key(versioned_provider, image_path)}.json"

    def _load_provider_cache(self, provider_name: str, image_path: Path) -> list[ReverseImageCandidate] | None:
        cache_path = self._provider_cache_path(provider_name, image_path)
        if not cache_path.exists():
            return None
        try:
            payload = load_json(cache_path)
            return [ReverseImageCandidate(**item) for item in payload]
        except Exception:
            return None

    def _save_provider_cache(self, provider_name: str, image_path: Path, candidates: list[ReverseImageCandidate]) -> None:
        cache_path = self._provider_cache_path(provider_name, image_path)
        save_json(cache_path, [asdict(candidate) for candidate in candidates])

    @staticmethod
    def _normalized_public_url(url: str | None) -> str:
        cleaned = str(url or "").strip()
        if not cleaned:
            return ""
        parsed = urlparse(cleaned)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}"
        return cleaned.rstrip("/").lower()

    def _sanitize_provider_candidates(
        self,
        candidates: list[ReverseImageCandidate],
        query_public_url: str | None,
    ) -> list[ReverseImageCandidate]:
        normalized_query_url = self._normalized_public_url(query_public_url)
        sanitized: list[ReverseImageCandidate] = []
        for candidate in candidates:
            metadata = dict(candidate.metadata or {})
            candidate_media_url = str(candidate.media_url or "").strip() or None
            candidate_query_url = str(metadata.get("query_image_url") or "").strip() or query_public_url

            media_matches_query = bool(
                candidate_media_url
                and (
                    self._normalized_public_url(candidate_media_url) == normalized_query_url
                    or self._normalized_public_url(candidate_media_url) == self._normalized_public_url(candidate_query_url)
                )
            )
            if media_matches_query:
                original_media_url = candidate_media_url
                fallback_media_url = None
                for fallback_candidate in (
                    metadata.get("thumbnail"),
                    metadata.get("provider_image_url"),
                ):
                    fallback_value = str(fallback_candidate or "").strip()
                    if not fallback_value:
                        continue
                    normalized_fallback = self._normalized_public_url(fallback_value)
                    if not normalized_fallback:
                        continue
                    if normalized_fallback == normalized_query_url:
                        continue
                    if normalized_fallback == self._normalized_public_url(candidate_query_url):
                        continue
                    fallback_media_url = fallback_value
                    break

                metadata["provider_media_url"] = original_media_url
                metadata["provider_media_sanitized"] = True
                candidate_media_url = fallback_media_url

            if query_public_url and "query_image_url" not in metadata:
                metadata["query_image_url"] = query_public_url

            sanitized.append(
                ReverseImageCandidate(
                    provider=candidate.provider,
                    page_url=candidate.page_url,
                    media_url=candidate_media_url,
                    title=candidate.title,
                    snippet=candidate.snippet,
                    timestamp=candidate.timestamp,
                    rank=candidate.rank,
                    frame_index=candidate.frame_index,
                    metadata=metadata,
                )
            )
        return sanitized

    def _public_query_url_cache_path(self, image_path: Path) -> Path:
        cache_key = reverse_query_cache_key("cloudinary_public_url", image_path)
        return self.public_query_url_dir / f"{cache_key}.json"

    def _ensure_public_query_url(self, image_path: Path) -> str | None:
        cache_path = self._public_query_url_cache_path(image_path)
        if cache_path.exists():
            try:
                payload = load_json(cache_path)
                cached_url = str(payload.get("url") or "").strip()
                if cached_url.startswith(("http://", "https://")):
                    return cached_url
            except Exception:
                pass

        try:
            public_url = upload_path_to_cloudinary(image_path, filename_override=image_path.name)
        except Exception:
            return None

        save_json(cache_path, {"url": public_url})
        return public_url

    def _is_low_signal_url(self, url: str | None) -> bool:
        if not url:
            return False
        netloc = urlparse(url).netloc.lower().removeprefix("www.")
        return any(netloc == domain or netloc.endswith(f".{domain}") for domain in self.low_signal_domains)

    def _filter_low_signal_records(self, records: list[OccurrenceRecord]) -> list[OccurrenceRecord]:
        return [record for record in records if not self._is_low_signal_url(record.url)]

    def _looks_generic(self, text: str | None) -> bool:
        lowered = str(text or "").lower()
        return any(keyword in lowered for keyword in self.generic_keywords)

    def _is_generic_domain(self, url: str | None) -> bool:
        if not url:
            return False
        netloc = urlparse(url).netloc.lower().removeprefix("www.")
        return any(netloc == domain or netloc.endswith(f".{domain}") for domain in self.generic_domains)

    def _specificity_boost(self, title: str | None) -> float:
        title_words = str(title or "").split()
        return 0.1 if len(title_words) > 6 else 0.0

    def _adjust_record_score(self, record: OccurrenceRecord) -> OccurrenceRecord:
        metadata = dict(record.metadata)
        num_sources = int(metadata.get("provider_hits") or metadata.get("reverse_provider_hits") or 1)
        num_frames = int(metadata.get("frame_hits") or metadata.get("reverse_frame_hits") or 1)

        raw_visual_similarity = float(record.visual_similarity) if record.visual_similarity is not None else None
        display_visual_similarity = raw_visual_similarity
        if display_visual_similarity is not None and num_sources == 1:
            display_visual_similarity = min(display_visual_similarity, 0.75)

        base_score = (
            display_visual_similarity
            if display_visual_similarity is not None
            else float(record.fused_similarity)
        )
        final_score = float(base_score)
        reasons: list[str] = []
        generic_penalty = 0.0
        generic_domain_penalty = 0.0
        source_penalty = 0.0
        frame_penalty = 0.0
        specificity_boost = 0.0

        generic_page_penalized = False
        if self._looks_generic(record.title) or self._looks_generic(record.caption):
            generic_penalty = 0.4
            final_score -= generic_penalty
            generic_page_penalized = True
            reasons.append("Generic page penalized")

        if self._is_generic_domain(record.url):
            generic_domain_penalty = 0.3
            final_score -= generic_domain_penalty
            generic_page_penalized = True
            reasons.append("Generic domain penalized")

        if num_sources < 2:
            source_penalty = 0.2
            final_score -= source_penalty
            reasons.append("Low source diversity")

        if num_frames < 2:
            frame_penalty = 0.1
            final_score -= frame_penalty
            reasons.append("Low frame coverage")

        specificity_boost = self._specificity_boost(record.title)
        if specificity_boost > 0:
            final_score += specificity_boost
            reasons.append("Specific title boost")

        if record.credibility_score >= 0.75:
            reasons.append("High similarity with trusted domain")
        elif generic_page_penalized:
            reasons.append("High similarity but generic content")
        elif num_sources < 2 or num_frames < 2:
            reasons.append("Moderate similarity, weak evidence")

        final_score = max(0.0, min(final_score, 1.0))
        metadata["raw_visual_similarity"] = raw_visual_similarity
        metadata["visual_similarity_display"] = display_visual_similarity
        metadata["final_score"] = final_score
        metadata["evidence_sources"] = num_sources
        metadata["evidence_frames"] = num_frames
        metadata["match_reason"] = "; ".join(dict.fromkeys(reasons)) if reasons else "Similarity score adjusted using evidence quality checks."
        metadata["score_components"] = {
            "similarity": raw_visual_similarity if raw_visual_similarity is not None else base_score,
            "source_trust": float(record.credibility_score),
            "evidence": min(1.0, (min(num_sources, 2) / 2.0) * 0.6 + (min(num_frames, 2) / 2.0) * 0.4),
            "generic_penalty": generic_penalty,
            "generic_domain_penalty": generic_domain_penalty,
            "source_penalty": source_penalty,
            "frame_penalty": frame_penalty,
            "specificity_boost": specificity_boost,
            "total_penalty": generic_penalty + generic_domain_penalty + source_penalty + frame_penalty,
            "total_boost": specificity_boost,
            "final_score": final_score,
        }

        return OccurrenceRecord(
            entry_id=record.entry_id,
            source_type=record.source_type,
            platform=record.platform,
            url=record.url,
            local_path=record.local_path,
            timestamp=record.timestamp,
            title=record.title,
            caption=record.caption,
            label=record.label,
            credibility_score=record.credibility_score,
            visual_similarity=display_visual_similarity,
            audio_similarity=record.audio_similarity,
            fused_similarity=final_score,
            context=record.context,
            context_scores=record.context_scores,
            is_mock=record.is_mock,
            metadata=metadata,
        )

    def _reverse_search_frame(self, image_path: Path, frame_index: int, max_results: int) -> list[ReverseImageCandidate]:
        candidates: list[ReverseImageCandidate] = []
        search_target = self._ensure_public_query_url(image_path) or image_path
        query_public_url = search_target if isinstance(search_target, str) and search_target.startswith(("http://", "https://")) else None
        for provider in self.reverse_providers:
            cached = self._load_provider_cache(provider.name, image_path)
            provider_candidates = cached if cached is not None and len(cached) >= max_results else []
            if cached is None or len(provider_candidates) < max_results:
                try:
                    provider_candidates = provider.search_image(search_target, max_results=max_results)
                except Exception:
                    provider_candidates = []
                provider_candidates = self._sanitize_provider_candidates(provider_candidates, query_public_url)
                if provider_candidates:
                    self._save_provider_cache(provider.name, image_path, provider_candidates)
            else:
                provider_candidates = self._sanitize_provider_candidates(provider_candidates, query_public_url)
            for candidate in provider_candidates:
                metadata = dict(candidate.metadata)
                metadata["frame_index"] = frame_index
                candidates.append(
                    ReverseImageCandidate(
                        provider=candidate.provider,
                        page_url=candidate.page_url,
                        media_url=candidate.media_url,
                        title=candidate.title,
                        snippet=candidate.snippet,
                        timestamp=candidate.timestamp,
                        rank=candidate.rank,
                        frame_index=frame_index,
                        metadata=metadata,
                    )
                )
        return candidates

    @staticmethod
    def _merge_reverse_candidates(candidates: list[ReverseImageCandidate]) -> list[dict[str, object]]:
        grouped: dict[str, dict[str, object]] = {}
        for candidate in candidates:
            key = candidate.key
            bucket = grouped.setdefault(
                key,
                {
                    "page_url": candidate.page_url,
                    "media_url": candidate.media_url,
                    "title": candidate.title,
                    "snippet": candidate.snippet,
                    "timestamp": candidate.timestamp,
                    "best_rank": candidate.rank or 9999,
                    "providers": set(),
                    "frame_indices": set(),
                    "raw_candidates": [],
                },
            )
            bucket["page_url"] = bucket["page_url"] or candidate.page_url
            bucket["media_url"] = bucket["media_url"] or candidate.media_url
            bucket["title"] = bucket["title"] or candidate.title
            bucket["snippet"] = bucket["snippet"] or candidate.snippet
            bucket["timestamp"] = bucket["timestamp"] or candidate.timestamp
            bucket["best_rank"] = min(int(bucket["best_rank"]), int(candidate.rank or 9999))
            bucket["providers"].add(candidate.provider)
            bucket["frame_indices"].add(candidate.frame_index)
            bucket["raw_candidates"].append(candidate)

        merged = []
        for key, bucket in grouped.items():
            merged.append(
                {
                    "candidate_key": key,
                    "page_url": bucket["page_url"],
                    "media_url": bucket["media_url"],
                    "title": bucket["title"],
                    "snippet": bucket["snippet"],
                    "timestamp": bucket["timestamp"],
                    "frame_hits": len(bucket["frame_indices"]),
                    "provider_hits": len(bucket["providers"]),
                    "providers": sorted(bucket["providers"]),
                    "best_rank": int(bucket["best_rank"]),
                    "raw_candidates": bucket["raw_candidates"],
                }
            )

        return sorted(
            merged,
            key=lambda item: (item["frame_hits"], item["provider_hits"], -item["best_rank"]),
            reverse=True,
        )

    def _occurrence_from_verified_candidate(
        self,
        candidate: dict[str, object],
        verification,
        resolved_media,
    ) -> OccurrenceRecord:
        page_url = str(candidate.get("page_url") or resolved_media.page_url or resolved_media.media_url)
        platform = infer_platform(page_url, fallback=resolved_media.platform)
        visual_similarity = float(verification.visual_similarity) if verification.visual_similarity is not None else None
        audio_similarity = float(verification.audio_similarity) if verification.audio_similarity is not None else None
        frame_hits = int(candidate.get("frame_hits") or 0)
        provider_hits = int(candidate.get("provider_hits") or 0)

        evidence_bits = []
        if visual_similarity is not None:
            evidence_bits.append(f"visual similarity {visual_similarity:.2f}")
        if audio_similarity is not None:
            evidence_bits.append(f"audio similarity {audio_similarity:.2f}")
        evidence_text = " and ".join(evidence_bits) if evidence_bits else "embedding verification"
        match_reason = (
            f"Verified through reverse-search candidate download using {evidence_text}. "
            f"Observed across {frame_hits} queried frame(s) and {provider_hits} reverse-search source(s)."
        )

        return OccurrenceRecord(
            entry_id=hashlib.sha1(f"{page_url}|{resolved_media.media_url or ''}".encode("utf-8")).hexdigest()[:16],
            source_type="external",
            platform=platform,
            url=page_url,
            timestamp=normalize_timestamp(resolved_media.timestamp),
            title=resolved_media.title or str(candidate.get("title") or page_url),
            caption=resolved_media.caption or str(candidate.get("snippet") or "").strip() or None,
            credibility_score=credibility_score_for_source(platform, page_url),
            visual_similarity=visual_similarity,
            audio_similarity=audio_similarity,
            fused_similarity=float(verification.combined_score),
            metadata={
                "provider": "reverse_search",
                "reverse_providers": list(candidate.get("providers") or []),
                "frame_hits": frame_hits,
                "provider_hits": provider_hits,
                "best_provider_rank": candidate.get("best_rank"),
                "media_url": resolved_media.media_url,
                "resolved_media_type": resolved_media.media_type,
                "downloaded_path": str(resolved_media.local_path) if resolved_media.local_path else None,
                "verification_visual_threshold": verification.metadata.get("visual_threshold"),
                "verification_audio_threshold": verification.metadata.get("audio_threshold"),
                "match_reason": match_reason,
                "match_type": verification.match_type,
                "confidence_label": verification.confidence_label,
                "hash_distance": verification.hash_distance,
                "embedding_score": verification.embedding_score,
                **resolved_media.metadata,
                **verification.metadata,
            },
        )

    def _verify_reverse_candidates(
        self,
        merged_candidates: list[dict[str, object]],
        original_visual_embedding: np.ndarray | None,
        original_audio_embedding: np.ndarray | None,
        original_media_path: str | Path | None,
        max_results: int,
    ) -> list[OccurrenceRecord]:
        verified: list[OccurrenceRecord] = []
        for candidate in merged_candidates:
            try:
                resolved = self.resolver.resolve(
                    page_url=str(candidate.get("page_url") or "") or None,
                    media_url=str(candidate.get("media_url") or "") or None,
                    title=str(candidate.get("title") or "") or None,
                    caption=str(candidate.get("snippet") or "") or None,
                    timestamp=str(candidate.get("timestamp") or "") or None,
                    metadata={"reverse_providers": list(candidate.get("providers") or [])},
                )
            except Exception:
                resolved = None

            if resolved is None or resolved.local_path is None or not resolved.local_path.exists():
                continue

            verification = self.verifier.verify_candidate(
                candidate_path=resolved.local_path,
                original_visual_embedding=original_visual_embedding,
                original_audio_embedding=original_audio_embedding,
                original_media_path=original_media_path,
            )
            if not verification.accepted:
                continue

            verified.append(self._occurrence_from_verified_candidate(candidate, verification, resolved))
            if len(verified) >= max_results:
                break

        return sorted(verified, key=lambda item: item.fused_similarity, reverse=True)[:max_results]

    def search(
        self,
        file_path: str | Path,
        query_hint: str | None,
        local_matches: list[dict[str, object]],
        max_results: int = 8,
        allow_mock_fallback: bool = False,
        original_visual_embedding: np.ndarray | None = None,
        original_audio_embedding: np.ndarray | None = None,
        source_url: str | None = None,
    ) -> list[OccurrenceRecord]:
        file_path = Path(file_path)

        reverse_verified: list[OccurrenceRecord] = []
        if self.reverse_providers and original_visual_embedding is not None:
            reverse_candidates: list[ReverseImageCandidate] = []
            public_source_url = self._public_source_url(source_url)
            if public_source_url:
                for provider in self.reverse_providers:
                    cached = self._load_provider_cache(provider.name, Path(file_path))
                    provider_candidates = cached if cached is not None else []
                    if cached is None:
                        try:
                            provider_candidates = provider.search_image(public_source_url, max_results=max_results)
                        except Exception:
                            provider_candidates = []
                        provider_candidates = self._sanitize_provider_candidates(provider_candidates, public_source_url)
                        if provider_candidates:
                            self._save_provider_cache(provider.name, Path(file_path), provider_candidates)
                    else:
                        provider_candidates = self._sanitize_provider_candidates(provider_candidates, public_source_url)
                    reverse_candidates.extend(provider_candidates)
            else:
                query_images = self._prepare_query_images(file_path)
                for frame_index, image_path in enumerate(query_images, start=1):
                    reverse_candidates.extend(self._reverse_search_frame(image_path, frame_index=frame_index, max_results=max_results))
            merged_candidates = self._merge_reverse_candidates(reverse_candidates)
            reverse_verified = self._verify_reverse_candidates(
                merged_candidates=merged_candidates,
                original_visual_embedding=original_visual_embedding,
                original_audio_embedding=original_audio_embedding,
                original_media_path=file_path,
                max_results=max_results,
            )

        if reverse_verified:
            filtered = self._filter_low_signal_records(reverse_verified)
            adjusted = [self._adjust_record_score(record) for record in filtered]
            adjusted.sort(key=lambda item: float(item.metadata.get("final_score", item.fused_similarity)), reverse=True)
            return adjusted[:max_results]

        fallback_results = self.query_fallback.search(
            query_hint=query_hint,
            local_matches=local_matches,
            max_results=max_results,
            allow_mock_fallback=allow_mock_fallback,
        )
        for result in fallback_results:
            result.metadata.setdefault("reverse_provider_status", self.reverse_provider_status)
            result.metadata.setdefault(
                "match_reason",
                "Returned by text-based fallback because verified reverse-search results were unavailable.",
            )
            result.metadata.setdefault("reverse_source_url_used", bool(source_url))
        filtered_fallback = self._filter_low_signal_records(fallback_results)
        adjusted_fallback = [self._adjust_record_score(record) for record in filtered_fallback]
        adjusted_fallback.sort(key=lambda item: float(item.metadata.get("final_score", item.fused_similarity)), reverse=True)
        return adjusted_fallback[:max_results]
