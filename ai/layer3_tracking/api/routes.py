from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import asdict
from threading import Lock
from time import monotonic
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ai.layer3_tracking.db.database import engine, get_db
from ai.layer3_tracking.db.schemas import HealthResponse, ReportResponse, TrackingResult
from ai.layer3_tracking.tracker.tracker import ContentNotFoundError, Layer2ClientError, TrackingService


LOGGER = logging.getLogger(__name__)
router = APIRouter()


class InMemoryRateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = Lock()
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = monotonic()
        with self._lock:
            bucket = self._requests[key]
            while bucket and now - bucket[0] > self.window_seconds:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please retry shortly.",
                )
            bucket.append(now)


RATE_LIMITER = InMemoryRateLimiter()


def get_tracking_service(request: Request) -> TrackingService:
    return request.app.state.tracking_service


def enforce_rate_limit(request: Request) -> None:
    client_host = request.client.host if request.client else "unknown"
    RATE_LIMITER.check(f"{client_host}:{request.url.path}")


@router.post("/track/{content_id}", response_model=TrackingResult)
def trigger_tracking(
    request: Request,
    content_id: UUID,
    _: None = Depends(enforce_rate_limit),
    tracking_service: TrackingService = Depends(get_tracking_service),
) -> TrackingResult:
    try:
        return tracking_service.track_content(content_id)
    except ContentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Layer2ClientError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        LOGGER.exception("Database error while tracking content=%s", content_id)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database error.") from exc


@router.get("/report/{content_id}", response_model=ReportResponse)
def get_report(
    request: Request,
    content_id: UUID,
    _: None = Depends(enforce_rate_limit),
    tracking_service: TrackingService = Depends(get_tracking_service),
) -> ReportResponse:
    try:
        return tracking_service.get_report(content_id)
    except ContentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        LOGGER.exception("Database error while reading report for content=%s", content_id)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database error.") from exc


@router.get("/health", response_model=HealthResponse)
def health(
    request: Request,
    _: None = Depends(enforce_rate_limit),
    session: Session = Depends(get_db),
) -> HealthResponse:
    try:
        session.execute(text("SELECT 1"))
        database_status = "ok"
    except SQLAlchemyError:
        database_status = "error"

    scheduler = getattr(request.app.state, "scheduler", None)
    scheduler_running = bool(scheduler.scheduler.running) if scheduler is not None else False
    last_scheduler_run = scheduler.last_run_completed_at if scheduler is not None else None
    tracking_service = get_tracking_service(request)
    api_usage = tracking_service.get_health_snapshot()
    metrics = tracking_service.metrics_collector.snapshot()
    overall_status = "ok" if database_status == "ok" and scheduler_running else "degraded"
    return HealthResponse(
        status=overall_status,
        database=database_status,
        scheduler_running=scheduler_running,
        last_scheduler_run=last_scheduler_run,
        api_usage=api_usage,
        metrics=asdict(metrics),
    )
