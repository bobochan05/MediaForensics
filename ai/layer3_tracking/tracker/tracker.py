from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import UUID

import requests
from requests import Response
from sqlalchemy.orm import Session, sessionmaker

from ai.layer3_tracking.db import crud
from ai.layer3_tracking.db.database import SessionLocal
from ai.layer3_tracking.db.models import Content, ContentStatus
from ai.layer3_tracking.services.api_limiter import ApiLimitDecision, ApiLimiter
from ai.layer3_tracking.services.metrics import MetricsCollector
from ai.layer3_tracking.services.risk_analyzer import RiskAnalysis, RiskAnalyzer
from ai.layer3_tracking.services.url_utils import extract_domain, is_trusted_domain, normalize_urls
from ai.layer3_tracking.tracker.comparator import ComparisonResult, compare_sources


LOGGER = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _extract_urls_from_layer2_payload(payload: dict[str, Any]) -> list[str]:
    urls: list[str] = []

    def collect(values: Any) -> None:
        if isinstance(values, list):
            for item in values:
                if isinstance(item, str):
                    urls.append(item)
                elif isinstance(item, dict):
                    for key in ("url", "media_url"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            urls.append(value)

    for key in ("urls", "sources", "similar_content", "matches", "exact_matches", "embedding_similar_content", "visually_similar_content", "related_content"):
        collect(payload.get(key))
    return urls


class Layer2ClientError(RuntimeError):
    pass


class ContentNotFoundError(RuntimeError):
    pass


class SupportsLayer2Fetch(Protocol):
    def fetch_urls(self, *, media_url: str | None, content_hash: str) -> list[str]:
        ...


class Layer2TrackingClient:
    """HTTP adapter for Layer 3 -> Layer 2 source discovery."""

    def __init__(self, endpoint: str | None = None, timeout_seconds: int | None = None) -> None:
        self.endpoint = endpoint or os.getenv("LAYER3_LAYER2_TRACK_ENDPOINT", "").strip()
        self.timeout_seconds = timeout_seconds or int(os.getenv("LAYER3_LAYER2_TIMEOUT_SECONDS", "30"))
        self.session = requests.Session()

    def _handle_response(self, response: Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise Layer2ClientError("Layer 2 returned a non-JSON response.") from exc

        if response.status_code >= 400:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            raise Layer2ClientError(f"Layer 2 tracking request failed: {detail or response.text}")
        if not isinstance(payload, dict):
            raise Layer2ClientError("Layer 2 returned an unexpected payload shape.")
        return payload

    def fetch_urls(self, *, media_url: str | None, content_hash: str) -> list[str]:
        if not self.endpoint:
            raise Layer2ClientError("LAYER3_LAYER2_TRACK_ENDPOINT is not configured.")

        payload = {
            "media_url": media_url,
            "hash": content_hash,
        }
        LOGGER.info("Calling Layer 2 tracking endpoint for hash=%s", content_hash)
        try:
            response = self.session.post(self.endpoint, json=payload, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise Layer2ClientError("Layer 2 request failed or timed out.") from exc

        body = self._handle_response(response)
        urls = _extract_urls_from_layer2_payload(body)
        return normalize_urls(urls)


@dataclass(slots=True)
class TrackContentResult:
    content_id: UUID
    total_sources: int
    new_sources: int
    growth_rate: float
    growth_velocity: float
    spread_score: float
    risk_score: float
    status: ContentStatus
    message: str


class TrackingService:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session] = SessionLocal,
        layer2_client: SupportsLayer2Fetch | None = None,
        api_limiter: ApiLimiter | None = None,
        risk_analyzer: RiskAnalyzer | None = None,
        metrics_collector: MetricsCollector | None = None,
        max_sources_per_run: int | None = None,
        retry_attempts: int | None = None,
        retry_backoff_seconds: float | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.layer2_client = layer2_client or Layer2TrackingClient()
        self.api_limiter = api_limiter or ApiLimiter(
            daily_limit=int(os.getenv("LAYER3_MAX_CALLS_PER_DAY", "5")),
            monthly_limit=int(os.getenv("LAYER3_MAX_CALLS_PER_MONTH", "250")),
            override_risk_threshold=float(os.getenv("LAYER3_RISK_OVERRIDE_THRESHOLD", "0.8")),
        )
        self.risk_analyzer = risk_analyzer or RiskAnalyzer(
            viral_source_threshold=int(os.getenv("LAYER3_VIRAL_SOURCE_THRESHOLD", "25"))
        )
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.max_sources_per_run = max_sources_per_run or int(os.getenv("LAYER3_MAX_SOURCES_PER_RUN", "5000"))
        self.retry_attempts = retry_attempts or int(os.getenv("LAYER3_RETRY_ATTEMPTS", "3"))
        self.retry_backoff_seconds = retry_backoff_seconds or float(os.getenv("LAYER3_RETRY_BACKOFF_SECONDS", "1"))
        self._lock_guard = threading.Lock()
        self._content_locks: dict[UUID, threading.Lock] = {}

    def _get_content_lock(self, content_id: UUID) -> threading.Lock:
        with self._lock_guard:
            lock = self._content_locks.get(content_id)
            if lock is None:
                lock = threading.Lock()
                self._content_locks[content_id] = lock
            return lock

    def _reserve_api_budget(self, session: Session, content: Content) -> ApiLimitDecision:
        LOGGER.info("Checking API budget for content=%s", content.id)
        return self.api_limiter.reserve_call(session, risk_score=content.risk_score)

    def _fetch_content(self, session: Session, content_id: UUID, *, for_update: bool = False) -> Content:
        content = crud.get_content(session, content_id, for_update=for_update)
        if content is None:
            raise ContentNotFoundError(f"Content {content_id} was not found.")
        return content

    def _call_layer2(self, content: Content) -> list[str]:
        if not content.media_url and not content.hash:
            raise Layer2ClientError("Content must provide media_url or hash for Layer 2 tracking.")

        for attempt in range(1, self.retry_attempts + 1):
            try:
                LOGGER.info(
                    "Calling Layer 2 for content=%s attempt=%s/%s",
                    content.id,
                    attempt,
                    self.retry_attempts,
                    extra={"event": "layer2_call_start", "extra_data": {"content_id": str(content.id), "attempt": attempt}},
                )
                urls = self.layer2_client.fetch_urls(media_url=content.media_url, content_hash=content.hash)
                if urls is None:
                    urls = []
                if not isinstance(urls, list):
                    raise Layer2ClientError("Layer 2 returned an invalid URL collection.")
                normalized = normalize_urls(urls)[: self.max_sources_per_run]
                LOGGER.info(
                    "Layer 2 returned %s normalized URLs for content=%s",
                    len(normalized),
                    content.id,
                    extra={"event": "layer2_call_success", "extra_data": {"content_id": str(content.id), "url_count": len(normalized)}},
                )
                return normalized
            except Exception as exc:
                LOGGER.warning(
                    "Layer 2 call failed for content=%s attempt=%s/%s",
                    content.id,
                    attempt,
                    self.retry_attempts,
                    extra={
                        "event": "layer2_call_failure",
                        "extra_data": {"content_id": str(content.id), "attempt": attempt, "error": str(exc)},
                    },
                )
                if attempt >= self.retry_attempts:
                    raise Layer2ClientError(f"Layer 2 request failed after {self.retry_attempts} attempts.") from exc
                time.sleep(self.retry_backoff_seconds * (2 ** (attempt - 1)))

    @staticmethod
    def _summarize_domains(urls: list[str]) -> tuple[int, int]:
        trusted = 0
        unknown = 0
        for url in urls:
            domain = extract_domain(url)
            if not domain:
                continue
            if is_trusted_domain(domain):
                trusted += 1
            else:
                unknown += 1
        return trusted, unknown

    def _create_failure_log(
        self,
        *,
        session: Session,
        content: Content,
        checked_at: datetime,
        reason: str,
    ) -> None:
        total_sources = crud.get_total_sources(session, content.id)
        crud.create_tracking_log(
            session,
            content_id=content.id,
            checked_at=checked_at,
            total_sources=total_sources,
            new_sources=0,
            growth_rate=0.0,
            growth_velocity=0.0,
            spread_score=content.risk_score,
            risk_score=content.risk_score,
            success=False,
            failure_reason=reason,
        )

    def _persist_tracking_result(
        self,
        *,
        session: Session,
        content: Content,
        checked_at: datetime,
        normalized_urls: list[str],
    ) -> tuple[ComparisonResult, RiskAnalysis, int]:
        existing_urls = crud.get_source_urls(session, content.id)
        comparison = compare_sources(existing_urls, normalized_urls)
        previous_log = crud.get_previous_tracking_log(session, content.id, success_only=True)

        LOGGER.info(
            "Comparing sources for content=%s previous_total=%s current_total=%s new=%s",
            content.id,
            comparison.previous_total,
            comparison.current_total,
            comparison.new_sources_count,
            extra={
                "event": "tracking_comparison",
                "extra_data": {
                    "content_id": str(content.id),
                    "previous_total": comparison.previous_total,
                    "current_total": comparison.current_total,
                    "new_sources": comparison.new_sources_count,
                },
            },
        )

        crud.touch_existing_sources(session, content.id, comparison.existing_sources, checked_at)
        crud.insert_new_sources(session, content.id, comparison.new_sources, checked_at)

        total_sources = crud.get_total_sources(session, content.id)
        trusted_domains, unknown_domains = self._summarize_domains(normalized_urls)
        analysis = self.risk_analyzer.analyze(
            previous_total=comparison.previous_total,
            new_sources=comparison.new_sources_count,
            total_sources=total_sources,
            checked_at=checked_at,
            previous_checked_at=previous_log.checked_at if previous_log is not None else None,
            trusted_domains=trusted_domains,
            unknown_domains=unknown_domains,
        )

        crud.update_content_tracking_state(
            session,
            content=content,
            last_checked=checked_at,
            risk_score=analysis.risk_score,
            status=analysis.status,
        )
        crud.create_tracking_log(
            session,
            content_id=content.id,
            checked_at=checked_at,
            total_sources=total_sources,
            new_sources=comparison.new_sources_count,
            growth_rate=analysis.growth_rate,
            growth_velocity=analysis.growth_velocity,
            spread_score=analysis.spread_score,
            risk_score=analysis.risk_score,
        )
        return comparison, analysis, total_sources

    def track_content(self, content_id: UUID) -> TrackContentResult:
        content_lock = self._get_content_lock(content_id)
        with content_lock:
            LOGGER.info(
                "Starting Layer 3 tracking for content=%s",
                content_id,
                extra={"event": "tracking_start", "extra_data": {"content_id": str(content_id)}},
            )
            checked_at = _utcnow()
            deferred_error: Layer2ClientError | None = None
            try:
                with self.session_factory.begin() as session:
                    content = self._fetch_content(session, content_id, for_update=True)
                    decision = self._reserve_api_budget(session, content)
                    if not decision.allowed:
                        LOGGER.info(
                            "Skipping tracking for content=%s reason=%s",
                            content_id,
                            decision.reason,
                            extra={
                                "event": "tracking_skipped",
                                "extra_data": {"content_id": str(content_id), "reason": decision.reason},
                            },
                        )
                        latest_log = crud.get_latest_tracking_log(session, content_id)
                        total_sources = latest_log.total_sources if latest_log is not None else crud.get_total_sources(session, content_id)
                        growth_rate = latest_log.growth_rate if latest_log is not None else 0.0
                        growth_velocity = latest_log.growth_velocity if latest_log is not None else 0.0
                        spread_score = latest_log.spread_score if latest_log is not None else content.risk_score
                        return TrackContentResult(
                            content_id=content_id,
                            total_sources=total_sources,
                            new_sources=0,
                            growth_rate=growth_rate,
                            growth_velocity=growth_velocity,
                            spread_score=spread_score,
                            risk_score=content.risk_score,
                            status=content.status,
                            message=f"Tracking skipped: {decision.reason}",
                        )

                    try:
                        normalized_urls = self._call_layer2(content)
                    except Layer2ClientError as exc:
                        self._create_failure_log(
                            session=session,
                            content=content,
                            checked_at=checked_at,
                            reason=str(exc),
                        )
                        self.metrics_collector.record_failure(api_calls_used=1)
                        LOGGER.error(
                            "Layer 2 tracking failed for content=%s",
                            content_id,
                            extra={"event": "tracking_failure", "extra_data": {"content_id": str(content_id), "error": str(exc)}},
                        )
                        deferred_error = exc
                    else:
                        comparison, analysis, total_sources = self._persist_tracking_result(
                            session=session,
                            content=content,
                            checked_at=checked_at,
                            normalized_urls=normalized_urls,
                        )
                        self.metrics_collector.record_success(
                            growth_rate=analysis.growth_rate,
                            growth_velocity=analysis.growth_velocity,
                            api_calls_used=1,
                        )
            except Layer2ClientError:
                raise
            except Exception as exc:
                self.metrics_collector.record_failure(api_calls_used=0)
                LOGGER.exception(
                    "Unexpected tracking failure for content=%s",
                    content_id,
                    extra={"event": "tracking_failure", "extra_data": {"content_id": str(content_id), "error": str(exc)}},
                )
                raise

            if deferred_error is not None:
                raise deferred_error

            LOGGER.info(
                "Finished Layer 3 tracking for content=%s total_sources=%s new_sources=%s status=%s",
                content_id,
                total_sources,
                comparison.new_sources_count,
                analysis.status.value,
                extra={
                    "event": "tracking_complete",
                    "extra_data": {
                        "content_id": str(content_id),
                        "total_sources": total_sources,
                        "new_sources": comparison.new_sources_count,
                        "growth_rate": analysis.growth_rate,
                        "growth_velocity": analysis.growth_velocity,
                        "spread_score": analysis.spread_score,
                        "status": analysis.status.value,
                    },
                },
            )
            return TrackContentResult(
                content_id=content_id,
                total_sources=total_sources,
                new_sources=comparison.new_sources_count,
                growth_rate=analysis.growth_rate,
                growth_velocity=analysis.growth_velocity,
                spread_score=analysis.spread_score,
                risk_score=analysis.risk_score,
                status=analysis.status,
                message="Tracking completed successfully.",
            )

    def get_report(self, content_id: UUID) -> dict[str, Any]:
        with self.session_factory() as session:
            content = self._fetch_content(session, content_id)
            history = crud.get_tracking_history(session, content_id)
            latest_log = history[-1] if history else None
            total_sources = latest_log.total_sources if latest_log is not None else crud.get_total_sources(session, content_id)
            new_sources = latest_log.new_sources if latest_log is not None else 0
            growth_rate = latest_log.growth_rate if latest_log is not None else 0.0
            growth_velocity = latest_log.growth_velocity if latest_log is not None else 0.0
            spread_score = latest_log.spread_score if latest_log is not None else content.risk_score
            return {
                "content_id": content.id,
                "total_sources": total_sources,
                "new_sources": new_sources,
                "growth_rate": growth_rate,
                "growth_velocity": growth_velocity,
                "spread_score": spread_score,
                "status": content.status,
                "history": [
                    {
                        "id": log.id,
                        "checked_at": log.checked_at,
                        "total_sources": log.total_sources,
                        "new_sources": log.new_sources,
                        "growth_rate": log.growth_rate,
                        "growth_velocity": log.growth_velocity,
                        "spread_score": log.spread_score,
                        "risk_score": log.risk_score,
                        "success": log.success,
                        "failure_reason": log.failure_reason,
                    }
                    for log in history
                ],
            }

    def get_health_snapshot(self) -> dict[str, int]:
        with self.session_factory() as session:
            return self.api_limiter.usage_snapshot(session)

    def track_all_content(self) -> list[TrackContentResult]:
        LOGGER.info("Starting scheduled Layer 3 tracking run")
        results: list[TrackContentResult] = []
        with self.session_factory() as session:
            content_ids = crud.list_content_ids(session)

        for content_id in content_ids:
            try:
                results.append(self.track_content(content_id))
            except Exception:
                LOGGER.exception("Tracking failed for content=%s", content_id)
        LOGGER.info("Scheduled Layer 3 run completed with %s tracked content items", len(results))
        return results


def track_content(content_id: UUID, service: TrackingService | None = None) -> TrackContentResult:
    tracker = service or TrackingService()
    return tracker.track_content(content_id)
