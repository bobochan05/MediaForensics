from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, selectinload

from ai.layer3_tracking.services.url_utils import extract_domain

from .models import ApiUsage, Content, ContentStatus, Source, TrackingLog


def create_content(
    session: Session,
    *,
    content_hash: str,
    media_url: str | None = None,
    risk_score: float = 0.0,
    status: ContentStatus = ContentStatus.stable,
) -> Content:
    content = Content(hash=content_hash, media_url=media_url, risk_score=risk_score, status=status)
    session.add(content)
    session.flush()
    return content


def get_content(session: Session, content_id: UUID, *, for_update: bool = False) -> Content | None:
    statement = (
        select(Content)
        .where(Content.id == content_id)
        .options(selectinload(Content.sources), selectinload(Content.tracking_logs))
    )
    if for_update:
        statement = statement.with_for_update()
    return session.scalars(statement).first()


def list_content_ids(session: Session) -> list[UUID]:
    return list(session.scalars(select(Content.id).order_by(Content.created_at.asc())))


def get_source_urls(session: Session, content_id: UUID) -> set[str]:
    return set(
        session.scalars(
            select(Source.url).where(Source.content_id == content_id)
        ).all()
    )


def touch_existing_sources(session: Session, content_id: UUID, urls: list[str], checked_at: datetime) -> None:
    if not urls:
        return
    session.execute(
        update(Source)
        .where(Source.content_id == content_id, Source.url.in_(urls))
        .values(last_seen=checked_at)
    )


def insert_new_sources(session: Session, content_id: UUID, urls: list[str], checked_at: datetime) -> int:
    if not urls:
        return 0

    rows = [
        {
            "content_id": content_id,
            "url": url,
            "domain": extract_domain(url),
            "first_seen": checked_at,
            "last_seen": checked_at,
        }
        for url in urls
    ]

    if session.bind and session.bind.dialect.name == "postgresql":
        statement = pg_insert(Source).values(rows).on_conflict_do_nothing(
            index_elements=["content_id", "url"]
        )
        result = session.execute(statement)
        return int(result.rowcount or 0)

    if session.bind and session.bind.dialect.name == "sqlite":
        statement = sqlite_insert(Source).values(rows).prefix_with("OR IGNORE")
        result = session.execute(statement)
        return int(result.rowcount or 0)

    inserted = 0
    for row in rows:
        session.add(Source(**row))
        session.flush()
        inserted += 1
    return inserted


def create_tracking_log(
    session: Session,
    *,
    content_id: UUID,
    checked_at: datetime,
    total_sources: int,
    new_sources: int,
    growth_rate: float,
    growth_velocity: float,
    spread_score: float,
    risk_score: float,
    success: bool = True,
    failure_reason: str | None = None,
) -> TrackingLog:
    tracking_log = TrackingLog(
        content_id=content_id,
        checked_at=checked_at,
        total_sources=total_sources,
        new_sources=new_sources,
        growth_rate=growth_rate,
        growth_velocity=growth_velocity,
        spread_score=spread_score,
        risk_score=risk_score,
        success=success,
        failure_reason=failure_reason,
    )
    session.add(tracking_log)
    session.flush()
    return tracking_log


def update_content_tracking_state(
    session: Session,
    *,
    content: Content,
    last_checked: datetime,
    risk_score: float,
    status: ContentStatus,
) -> Content:
    content.last_checked = last_checked
    content.risk_score = risk_score
    content.status = status
    session.flush()
    return content


def get_latest_tracking_log(session: Session, content_id: UUID) -> TrackingLog | None:
    statement = (
        select(TrackingLog)
        .where(TrackingLog.content_id == content_id)
        .order_by(TrackingLog.checked_at.desc())
        .limit(1)
    )
    return session.scalars(statement).first()


def get_tracking_history(session: Session, content_id: UUID) -> list[TrackingLog]:
    statement = (
        select(TrackingLog)
        .where(TrackingLog.content_id == content_id)
        .order_by(TrackingLog.checked_at.asc())
    )
    return list(session.scalars(statement).all())


def get_previous_tracking_log(
    session: Session,
    content_id: UUID,
    before: datetime | None = None,
    *,
    success_only: bool = False,
) -> TrackingLog | None:
    statement = select(TrackingLog).where(TrackingLog.content_id == content_id)
    if before is not None:
        statement = statement.where(TrackingLog.checked_at < before)
    if success_only:
        statement = statement.where(TrackingLog.success.is_(True))
    statement = statement.order_by(TrackingLog.checked_at.desc()).limit(1)
    return session.scalars(statement).first()


def get_or_create_api_usage(session: Session, usage_date: date, *, for_update: bool = False) -> ApiUsage:
    statement = select(ApiUsage).where(ApiUsage.date == usage_date)
    if for_update:
        statement = statement.with_for_update()
    usage = session.scalars(statement).first()
    if usage is None:
        usage = ApiUsage(date=usage_date, calls_made=0)
        session.add(usage)
        session.flush()
    return usage


def get_monthly_calls(session: Session, target_date: date) -> int:
    first_day = target_date.replace(day=1)
    total = session.scalar(
        select(func.coalesce(func.sum(ApiUsage.calls_made), 0)).where(
            ApiUsage.date >= first_day,
            ApiUsage.date <= target_date,
        )
    )
    return int(total or 0)


def increment_api_usage(session: Session, usage: ApiUsage, calls: int = 1) -> ApiUsage:
    usage.calls_made += calls
    session.flush()
    return usage


def get_total_sources(session: Session, content_id: UUID) -> int:
    total = session.scalar(select(func.count(Source.id)).where(Source.content_id == content_id))
    return int(total or 0)


def get_daily_api_usage(session: Session, usage_date: date) -> int:
    usage = session.scalars(select(ApiUsage).where(ApiUsage.date == usage_date)).first()
    return int(usage.calls_made if usage is not None else 0)
