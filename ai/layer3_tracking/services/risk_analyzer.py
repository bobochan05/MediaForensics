from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ai.layer3_tracking.db.models import ContentStatus


@dataclass(slots=True)
class RiskAnalysis:
    growth_rate: float
    growth_velocity: float
    domain_weight: float
    spread_score: float
    risk_score: float
    status: ContentStatus


class RiskAnalyzer:
    def __init__(
        self,
        *,
        viral_source_threshold: int = 25,
        slow_spread_growth_threshold: float = 0.15,
        spreading_growth_threshold: float = 0.5,
        viral_growth_threshold: float = 1.0,
    ) -> None:
        self.viral_source_threshold = viral_source_threshold
        self.slow_spread_growth_threshold = slow_spread_growth_threshold
        self.spreading_growth_threshold = spreading_growth_threshold
        self.viral_growth_threshold = viral_growth_threshold

    @staticmethod
    def _hours_between(current_checked_at: datetime, previous_checked_at: datetime | None) -> float:
        if previous_checked_at is None:
            return 1.0
        if current_checked_at.tzinfo is not None and previous_checked_at.tzinfo is None:
            previous_checked_at = previous_checked_at.replace(tzinfo=timezone.utc)
        elif current_checked_at.tzinfo is None and previous_checked_at.tzinfo is not None:
            current_checked_at = current_checked_at.replace(tzinfo=timezone.utc)
        hours = (current_checked_at - previous_checked_at).total_seconds() / 3600.0
        return hours if hours > 0 else 1.0

    @staticmethod
    def compute_growth_rate(previous_total: int, new_sources: int) -> float:
        if previous_total <= 0:
            return float(new_sources) if new_sources > 0 else 0.0
        return float(new_sources / previous_total)

    @staticmethod
    def compute_domain_weight(*, trusted_domains: int, unknown_domains: int, total_sources: int) -> float:
        if total_sources <= 0:
            return 0.0
        weighted_domains = trusted_domains + (0.35 * unknown_domains)
        return round(min(weighted_domains / total_sources, 1.0), 4)

    def classify_status(self, *, growth_rate: float, growth_velocity: float, total_sources: int, spread_score: float) -> ContentStatus:
        if (
            growth_rate > self.viral_growth_threshold
            or total_sources > self.viral_source_threshold
            or spread_score >= 0.85
        ):
            return ContentStatus.viral
        if growth_rate > self.spreading_growth_threshold or growth_velocity >= 0.2 or spread_score >= 0.6:
            return ContentStatus.spreading
        if growth_rate > self.slow_spread_growth_threshold or growth_velocity > 0:
            return ContentStatus.slow_spread
        return ContentStatus.stable

    def compute_growth_velocity(
        self,
        *,
        growth_rate: float,
        checked_at: datetime,
        previous_checked_at: datetime | None,
    ) -> float:
        hours = self._hours_between(checked_at, previous_checked_at)
        return round(growth_rate / hours, 4)

    def compute_spread_score(
        self,
        *,
        growth_rate: float,
        growth_velocity: float,
        domain_weight: float,
        total_sources: int,
        new_sources: int,
    ) -> float:
        growth_component = min(growth_rate / max(self.viral_growth_threshold, 1.0), 1.0)
        velocity_component = min(growth_velocity / 0.5, 1.0)
        source_component = min(total_sources / max(self.viral_source_threshold, 1), 1.0)
        novelty_component = min(new_sources / 10.0, 1.0)
        score = (
            (0.3 * growth_component)
            + (0.25 * velocity_component)
            + (0.2 * source_component)
            + (0.15 * novelty_component)
            + (0.1 * (1.0 - domain_weight))
        )
        return round(min(score, 1.0), 4)

    def analyze(
        self,
        *,
        previous_total: int,
        new_sources: int,
        total_sources: int,
        checked_at: datetime,
        previous_checked_at: datetime | None,
        trusted_domains: int,
        unknown_domains: int,
    ) -> RiskAnalysis:
        growth_rate = self.compute_growth_rate(previous_total, new_sources)
        growth_velocity = self.compute_growth_velocity(
            growth_rate=growth_rate,
            checked_at=checked_at,
            previous_checked_at=previous_checked_at,
        )
        domain_weight = self.compute_domain_weight(
            trusted_domains=trusted_domains,
            unknown_domains=unknown_domains,
            total_sources=total_sources,
        )
        spread_score = self.compute_spread_score(
            growth_rate=growth_rate,
            growth_velocity=growth_velocity,
            domain_weight=domain_weight,
            total_sources=total_sources,
            new_sources=new_sources,
        )
        status = self.classify_status(
            growth_rate=growth_rate,
            growth_velocity=growth_velocity,
            total_sources=total_sources,
            spread_score=spread_score,
        )
        return RiskAnalysis(
            growth_rate=growth_rate,
            growth_velocity=growth_velocity,
            domain_weight=domain_weight,
            spread_score=spread_score,
            risk_score=spread_score,
            status=status,
        )
