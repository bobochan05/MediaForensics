from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GrowthSnapshot:
    previous_total: int
    current_total: int
    growth_rate_percent: float


def compute_growth(previous_total: int, current_total: int) -> GrowthSnapshot:
    baseline = max(1, int(previous_total))
    growth_rate = ((int(current_total) - int(previous_total)) / baseline) * 100.0
    return GrowthSnapshot(
        previous_total=int(previous_total),
        current_total=int(current_total),
        growth_rate_percent=round(growth_rate, 2),
    )

