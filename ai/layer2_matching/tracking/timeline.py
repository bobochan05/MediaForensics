from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from statistics import mean
from typing import Sequence

from ai.layer2_matching.tracking.metadata_parser import OccurrenceRecord, normalize_timestamp


def build_spread_timeline(occurrences: Sequence[OccurrenceRecord]) -> list[dict[str, float | int | str]]:
    buckets: "OrderedDict[str, int]" = OrderedDict()
    for occurrence in sorted(occurrences, key=lambda item: item.timestamp or ""):
        timestamp = normalize_timestamp(occurrence.timestamp)
        if timestamp is None:
            continue
        day_key = timestamp[:10]
        buckets[day_key] = buckets.get(day_key, 0) + 1

    timeline: list[dict[str, float | int | str]] = []
    cumulative = 0
    previous_count = 0
    previous_counts: list[int] = []
    for day_key, count in buckets.items():
        cumulative += count
        velocity = float(max(count - previous_count, 0))
        baseline = mean(previous_counts) if previous_counts else float(count)
        spike_score = float(count / baseline) if baseline > 0 else 1.0
        timeline.append(
            {
                "timestamp": f"{day_key}T00:00:00+00:00",
                "count": int(count),
                "cumulative_count": int(cumulative),
                "velocity": velocity,
                "spike_score": spike_score,
            }
        )
        previous_count = count
        previous_counts.append(count)
    return timeline


def estimate_origin(occurrences: Sequence[OccurrenceRecord]) -> dict[str, str | None]:
    dated_occurrences = [item for item in occurrences if normalize_timestamp(item.timestamp) is not None]
    if not dated_occurrences:
        return {
            "timestamp": None,
            "source": None,
            "url": None,
            "platform": None,
            "note": "No reliable timestamp was available, so the probable origin could not be estimated.",
        }

    earliest = min(
        dated_occurrences,
        key=lambda item: datetime.fromisoformat(normalize_timestamp(item.timestamp) or "9999-12-31T00:00:00+00:00"),
    )
    return {
        "timestamp": normalize_timestamp(earliest.timestamp),
        "source": earliest.title or earliest.local_path or earliest.url,
        "url": earliest.url,
        "platform": earliest.platform,
        "note": "Earliest accessible occurrence is used as a proxy for probable origin.",
    }
