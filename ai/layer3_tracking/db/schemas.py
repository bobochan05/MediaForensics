from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .models import ContentStatus


class SourceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    url: str
    domain: str
    first_seen: datetime
    last_seen: datetime


class TrackingLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    checked_at: datetime
    total_sources: int
    new_sources: int
    growth_rate: float
    growth_velocity: float
    spread_score: float
    risk_score: float
    success: bool
    failure_reason: str | None = None


class ContentCreate(BaseModel):
    hash: str = Field(..., min_length=1, max_length=128)
    media_url: str | None = None
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ContentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    hash: str
    media_url: str | None
    created_at: datetime
    last_checked: datetime | None
    risk_score: float
    status: ContentStatus


class TrackingResult(BaseModel):
    content_id: UUID
    total_sources: int
    new_sources: int
    growth_rate: float
    growth_velocity: float
    spread_score: float
    risk_score: float
    status: ContentStatus
    message: str


class ApiUsageHealth(BaseModel):
    daily_calls: int
    daily_limit: int
    monthly_calls: int
    monthly_limit: int


class MetricsSnapshot(BaseModel):
    total_tracking_runs: int
    failed_tracking_runs: int
    api_calls_used: int
    average_growth_rate: float
    average_growth_velocity: float


class ReportResponse(BaseModel):
    content_id: UUID
    total_sources: int
    new_sources: int
    growth_rate: float
    growth_velocity: float
    spread_score: float
    status: ContentStatus
    history: list[TrackingLogRead]


class HealthResponse(BaseModel):
    status: str
    database: str
    scheduler_running: bool
    last_scheduler_run: datetime | None = None
    api_usage: ApiUsageHealth
    metrics: MetricsSnapshot
