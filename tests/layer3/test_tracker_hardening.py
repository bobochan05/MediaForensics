from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ai.layer3_tracking.db import crud
from ai.layer3_tracking.db.database import Base
from ai.layer3_tracking.db.models import ContentStatus
from ai.layer3_tracking.services.api_limiter import ApiLimiter
from ai.layer3_tracking.services.risk_analyzer import RiskAnalyzer
from ai.layer3_tracking.tracker.tracker import Layer2ClientError, TrackingService


class StaticLayer2Client:
    def __init__(self, response: list[str]) -> None:
        self.response = response

    def fetch_urls(self, *, media_url: str | None, content_hash: str) -> list[str]:
        return list(self.response)


class FailingLayer2Client:
    def __init__(self, failures_before_success: int = 3) -> None:
        self.failures_before_success = failures_before_success
        self.calls = 0

    def fetch_urls(self, *, media_url: str | None, content_hash: str) -> list[str]:
        self.calls += 1
        raise Layer2ClientError("mock failure")


def build_sqlite_service(layer2_client, *, db_path: str | None = None, max_sources_per_run: int = 5000) -> tuple[TrackingService, UUID, sessionmaker]:
    if db_path is None:
        engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    else:
        engine = create_engine(
            f"sqlite+pysqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            future=True,
        )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

    with SessionLocal.begin() as session:
        content = crud.create_content(
            session,
            content_hash="hardening-hash",
            media_url="https://example.com/media.jpg",
            risk_score=0.2,
            status=ContentStatus.stable,
        )
        content_id = content.id

    service = TrackingService(
        session_factory=SessionLocal,
        layer2_client=layer2_client,
        api_limiter=ApiLimiter(daily_limit=50, monthly_limit=500),
        risk_analyzer=RiskAnalyzer(viral_source_threshold=10),
        max_sources_per_run=max_sources_per_run,
        retry_backoff_seconds=0.01,
    )
    return service, content_id, SessionLocal


def test_tracker_is_idempotent_for_same_sources():
    service, content_id, SessionLocal = build_sqlite_service(
        StaticLayer2Client(
            [
                "https://example.com/a?utm_source=test",
                "https://example.com/a",
                "https://example.com/b",
            ]
        )
    )

    first = service.track_content(content_id)
    second = service.track_content(content_id)

    assert first.total_sources == 2
    assert second.total_sources == 2
    assert second.new_sources == 0

    with SessionLocal() as session:
        assert crud.get_total_sources(session, content_id) == 2
        history = crud.get_tracking_history(session, content_id)
        assert len(history) == 2
        assert all(log.success for log in history)


def test_tracker_logs_failure_without_corrupting_state():
    service, content_id, SessionLocal = build_sqlite_service(FailingLayer2Client())

    with pytest.raises(Layer2ClientError):
        service.track_content(content_id)

    with SessionLocal() as session:
        assert crud.get_total_sources(session, content_id) == 0
        history = crud.get_tracking_history(session, content_id)
        assert len(history) == 1
        assert history[0].success is False
        assert history[0].failure_reason is not None


def test_tracker_handles_empty_results():
    service, content_id, SessionLocal = build_sqlite_service(StaticLayer2Client([]))

    result = service.track_content(content_id)

    assert result.total_sources == 0
    assert result.new_sources == 0
    with SessionLocal() as session:
        history = crud.get_tracking_history(session, content_id)
        assert len(history) == 1
        assert history[0].success is True


def test_tracker_handles_large_unique_source_sets():
    urls = [f"https://example.com/post/{index}?utm_source=test" for index in range(1200)]
    urls.extend(["invalid-url", "https://example.com/post/1"])
    service, content_id, SessionLocal = build_sqlite_service(StaticLayer2Client(urls), max_sources_per_run=1500)

    result = service.track_content(content_id)

    assert result.total_sources == 1200
    with SessionLocal() as session:
        assert crud.get_total_sources(session, content_id) == 1200


def test_tracker_serializes_concurrent_runs_for_same_content():
    tmp_dir = Path("layer3/tests/_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_path = str((tmp_dir / f"{uuid4().hex}.db").resolve())
    service, content_id, SessionLocal = build_sqlite_service(
        StaticLayer2Client(
            [
                "https://example.com/a",
                "https://example.com/b",
            ]
        ),
        db_path=db_path,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(service.track_content, [content_id, content_id]))

        assert len(results) == 2
        assert sorted(result.total_sources for result in results) == [2, 2]
        with SessionLocal() as session:
            assert crud.get_total_sources(session, content_id) == 2
            history = crud.get_tracking_history(session, content_id)
            assert len(history) == 2
            assert history[0].new_sources == 2
            assert history[1].new_sources == 0
    finally:
        pass
