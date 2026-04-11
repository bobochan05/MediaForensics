from __future__ import annotations

from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ai.layer3_tracking.db import crud
from ai.layer3_tracking.db.database import Base
from ai.layer3_tracking.db.models import ContentStatus
from ai.layer3_tracking.services.api_limiter import ApiLimiter
from ai.layer3_tracking.services.risk_analyzer import RiskAnalyzer
from ai.layer3_tracking.tracker.tracker import TrackingService


class MockLayer2Client:
    def __init__(self) -> None:
        self.responses = [
            [
                "https://news.example.com/article-1?utm_source=test",
                "https://x.com/example/status/1",
                "not-a-valid-url",
            ],
            [
                "https://news.example.com/article-1?utm_source=test",
                "https://x.com/example/status/1",
                "https://reddit.com/r/test/comments/123",
                "https://reddit.com/r/test/comments/123?ref=campaign",
            ],
        ]
        self.calls = 0

    def fetch_urls(self, *, media_url: str | None, content_hash: str) -> list[str]:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def build_tracking_service() -> tuple[TrackingService, UUID]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

    with SessionLocal.begin() as session:
        content = crud.create_content(
            session,
            content_hash="mock-hash",
            media_url="https://example.com/media.jpg",
            risk_score=0.1,
            status=ContentStatus.stable,
        )
        content_id = content.id

    service = TrackingService(
        session_factory=SessionLocal,
        layer2_client=MockLayer2Client(),
        api_limiter=ApiLimiter(daily_limit=20, monthly_limit=500),
        risk_analyzer=RiskAnalyzer(viral_source_threshold=4),
    )
    return service, content_id


def test_tracking_flow_appends_history_and_detects_growth():
    service, content_id = build_tracking_service()

    first = service.track_content(content_id)
    second = service.track_content(content_id)
    report = service.get_report(content_id)

    assert first.total_sources == 2
    assert first.new_sources == 2
    assert second.total_sources == 3
    assert second.new_sources == 1
    assert second.growth_rate == 0.5
    assert second.growth_velocity >= 0.0
    assert second.spread_score >= 0.0
    assert report["total_sources"] == 3
    assert len(report["history"]) == 2
    assert report["history"][-1]["success"] is True
