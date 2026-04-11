from __future__ import annotations

import enum
import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from .database import Base


class ContentStatus(str, enum.Enum):
    stable = "stable"
    slow_spread = "slow_spread"
    spreading = "spreading"
    viral = "viral"


class Content(Base):
    __tablename__ = "content"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    hash: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    media_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_checked: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    status: Mapped[ContentStatus] = mapped_column(
        Enum(ContentStatus, name="content_status"),
        default=ContentStatus.stable,
        nullable=False,
    )

    sources: Mapped[list["Source"]] = relationship(back_populates="content", cascade="all, delete-orphan")
    tracking_logs: Mapped[list["TrackingLog"]] = relationship(back_populates="content", cascade="all, delete-orphan")


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (UniqueConstraint("content_id", "url", name="uq_sources_content_url"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("content.id", ondelete="CASCADE"), nullable=False, index=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    content: Mapped[Content] = relationship(back_populates="sources")


class TrackingLog(Base):
    __tablename__ = "tracking_logs"
    __table_args__ = (
        Index("ix_tracking_logs_content_id_checked_at", "content_id", "checked_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("content.id", ondelete="CASCADE"), nullable=False, index=True)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    total_sources: Mapped[int] = mapped_column(Integer, nullable=False)
    new_sources: Mapped[int] = mapped_column(Integer, nullable=False)
    growth_rate: Mapped[float] = mapped_column(Float, nullable=False)
    growth_velocity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    spread_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="1")
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    content: Mapped[Content] = relationship(back_populates="tracking_logs")


class ApiUsage(Base):
    __tablename__ = "api_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, unique=True, nullable=False, index=True)
    calls_made: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
