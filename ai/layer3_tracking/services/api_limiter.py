from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from ai.layer3_tracking.db import crud


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ApiLimitDecision:
    allowed: bool
    override_used: bool
    calls_today: int
    monthly_calls: int
    reason: str


class ApiLimiter:
    def __init__(
        self,
        *,
        daily_limit: int = 5,
        monthly_limit: int = 250,
        override_risk_threshold: float = 0.8,
    ) -> None:
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.override_risk_threshold = override_risk_threshold

    def reserve_call(self, session: Session, *, risk_score: float) -> ApiLimitDecision:
        today = datetime.now(UTC).date()
        usage = crud.get_or_create_api_usage(session, today, for_update=True)
        monthly_calls = crud.get_monthly_calls(session, today)

        if monthly_calls >= self.monthly_limit:
            LOGGER.warning("Layer 3 API limit reached for month: %s/%s", monthly_calls, self.monthly_limit)
            return ApiLimitDecision(False, False, usage.calls_made, monthly_calls, "monthly_limit_reached")

        override_allowed = risk_score > self.override_risk_threshold
        if usage.calls_made >= self.daily_limit and not override_allowed:
            LOGGER.info("Layer 3 daily API limit reached: %s/%s", usage.calls_made, self.daily_limit)
            return ApiLimitDecision(False, False, usage.calls_made, monthly_calls, "daily_limit_reached")

        crud.increment_api_usage(session, usage, calls=1)
        LOGGER.info(
            "Reserved Layer 3 API call (today=%s, month=%s, override=%s)",
            usage.calls_made,
            monthly_calls + 1,
            override_allowed and usage.calls_made > self.daily_limit,
        )
        return ApiLimitDecision(
            True,
            override_allowed and usage.calls_made > self.daily_limit,
            usage.calls_made,
            monthly_calls + 1,
            "allowed",
        )

    def usage_snapshot(self, session: Session) -> dict[str, int]:
        today = datetime.now(UTC).date()
        return {
            "daily_calls": crud.get_daily_api_usage(session, today),
            "daily_limit": self.daily_limit,
            "monthly_calls": crud.get_monthly_calls(session, today),
            "monthly_limit": self.monthly_limit,
        }
