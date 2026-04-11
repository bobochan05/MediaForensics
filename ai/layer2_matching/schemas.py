from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SimilarContentItem(BaseModel):
    id: str
    url: str | None = None
    media_url: str | None = None
    local_path: str | None = None
    visual_similarity: float | None = None
    audio_similarity: float | None = None
    fused_similarity: float
    combined_score: float | None = None
    platform: str
    timestamp: str | None = None
    source_type: str
    title: str | None = None
    caption: str | None = None
    context: str
    context_scores: dict[str, float] = Field(default_factory=dict)
    credibility_score: float
    is_mock: bool = False
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimelinePoint(BaseModel):
    timestamp: str
    count: int
    cumulative_count: int
    velocity: float
    spike_score: float


class OriginEstimate(BaseModel):
    timestamp: str | None = None
    source: str | None = None
    url: str | None = None
    platform: str | None = None
    note: str


class RiskBreakdown(BaseModel):
    fake_probability: float
    occurrence_score: float
    spread_velocity_score: float
    credibility_risk: float
    misuse_probability: float
    visual_similarity_score: float
    audio_similarity_score: float
    final_score: float


class AnalysisResponse(BaseModel):
    analysis_id: str
    filename: str
    media_type: str
    is_fake: bool
    confidence: float
    similar_content: list[SimilarContentItem]
    matches: list[SimilarContentItem] = Field(default_factory=list)
    origin_estimate: OriginEstimate
    spread_timeline: list[TimelinePoint]
    risk_score: float
    risk_level: str
    risk_breakdown: RiskBreakdown
    confidence_explanation: str
    visual_embedding_dim: int
    audio_embedding_dim: int | None = None
    external_matches_used: bool
    created_at: str
    layer2_insights: dict[str, Any] = Field(default_factory=dict)
