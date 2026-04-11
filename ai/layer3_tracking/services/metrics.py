from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(slots=True)
class MetricsSnapshotData:
    total_tracking_runs: int
    failed_tracking_runs: int
    api_calls_used: int
    average_growth_rate: float
    average_growth_velocity: float


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = Lock()
        self._total_tracking_runs = 0
        self._failed_tracking_runs = 0
        self._api_calls_used = 0
        self._growth_rate_sum = 0.0
        self._growth_velocity_sum = 0.0

    def record_success(self, *, growth_rate: float, growth_velocity: float, api_calls_used: int = 1) -> None:
        with self._lock:
            self._total_tracking_runs += 1
            self._api_calls_used += api_calls_used
            self._growth_rate_sum += growth_rate
            self._growth_velocity_sum += growth_velocity

    def record_failure(self, *, api_calls_used: int = 0) -> None:
        with self._lock:
            self._total_tracking_runs += 1
            self._failed_tracking_runs += 1
            self._api_calls_used += api_calls_used

    def snapshot(self) -> MetricsSnapshotData:
        with self._lock:
            successful_runs = self._total_tracking_runs - self._failed_tracking_runs
            average_growth_rate = self._growth_rate_sum / successful_runs if successful_runs > 0 else 0.0
            average_growth_velocity = self._growth_velocity_sum / successful_runs if successful_runs > 0 else 0.0
            return MetricsSnapshotData(
                total_tracking_runs=self._total_tracking_runs,
                failed_tracking_runs=self._failed_tracking_runs,
                api_calls_used=self._api_calls_used,
                average_growth_rate=round(average_growth_rate, 4),
                average_growth_velocity=round(average_growth_velocity, 4),
            )
