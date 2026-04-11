from __future__ import annotations

from pathlib import Path
import sys
from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.layer3_tracking.db import crud
from ai.layer3_tracking.db.database import Base
from ai.layer3_tracking.db.models import ContentStatus
from ai.layer3_tracking.services.api_limiter import ApiLimiter
from ai.layer3_tracking.services.risk_analyzer import RiskAnalyzer
from ai.layer3_tracking.tracker.tracker import TrackingService


class MockLayer2Client:
    def __init__(self) -> None:
        self.responses = [
            ["https://news.example.com/post-1", "https://x.com/example/status/1"],
            ["https://news.example.com/post-1", "https://x.com/example/status/1", "https://reddit.com/r/test/comments/123"],
        ]
        self.index = 0

    def fetch_urls(self, *, media_url: str | None, content_hash: str) -> list[str]:
        response = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return response


def build_service() -> tuple[TrackingService, UUID]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

    with SessionLocal.begin() as session:
        content = crud.create_content(
            session,
            content_hash="demo-hash-123",
            media_url="https://example.com/deepfake.jpg",
            risk_score=0.2,
            status=ContentStatus.stable,
        )
        content_id = content.id

    service = TrackingService(
        session_factory=SessionLocal,
        layer2_client=MockLayer2Client(),
        api_limiter=ApiLimiter(daily_limit=10, monthly_limit=100),
        risk_analyzer=RiskAnalyzer(viral_source_threshold=5),
    )
    return service, content_id


if __name__ == "__main__":
    service, content_id = build_service()
    first = service.track_content(content_id)
    second = service.track_content(content_id)
    report = service.get_report(content_id)

    print("First run:", first)
    print("Second run:", second)
    print("Report:", report)
