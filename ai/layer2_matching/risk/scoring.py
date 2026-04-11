from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ai.layer2_matching.tracking.metadata_parser import OccurrenceRecord


@dataclass
class RiskAssessment:
    score: float
    level: str
    breakdown: dict[str, float]
    explanation: str


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_risk_assessment(
    fake_probability: float,
    occurrences: Sequence[OccurrenceRecord],
    timeline: Sequence[dict[str, float | int | str]],
) -> RiskAssessment:
    occurrence_score = _clamp_01(len(occurrences) / 12.0)
    spread_velocity = max((float(point["velocity"]) for point in timeline), default=0.0)
    spread_velocity_score = _clamp_01(spread_velocity / 5.0)
    average_credibility = sum(item.credibility_score for item in occurrences) / len(occurrences) if occurrences else 0.5
    credibility_risk = _clamp_01(1.0 - average_credibility)
    misuse_probability = max((item.context_scores.get("propaganda / misinformation", 0.0) for item in occurrences), default=0.0)
    visual_similarity_score = max((float(item.visual_similarity or 0.0) for item in occurrences), default=0.0)
    audio_similarity_score = max((float(item.audio_similarity or 0.0) for item in occurrences), default=0.0)

    weights = {
        "fake_probability": 0.25,
        "occurrence_score": 0.15,
        "spread_velocity_score": 0.15,
        "credibility_risk": 0.10,
        "misuse_probability": 0.10,
        "visual_similarity_score": 0.15,
        "audio_similarity_score": 0.10,
    }
    breakdown = {
        "fake_probability": _clamp_01(fake_probability),
        "occurrence_score": occurrence_score,
        "spread_velocity_score": spread_velocity_score,
        "credibility_risk": credibility_risk,
        "misuse_probability": _clamp_01(misuse_probability),
        "visual_similarity_score": _clamp_01(visual_similarity_score),
        "audio_similarity_score": _clamp_01(audio_similarity_score),
    }
    final_score = round(10.0 * sum(weights[key] * breakdown[key] for key in weights), 2)

    if final_score >= 7.5:
        level = "HIGH"
    elif final_score >= 4.5:
        level = "MEDIUM"
    else:
        level = "LOW"

    ranked_factors = sorted(
        ((name, weights[name] * value) for name, value in breakdown.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    explanation_names = [name.replace("_", " ") for name, _ in ranked_factors[:3]]
    explanation = (
        f"Risk is {level.lower()} primarily because of "
        f"{', '.join(explanation_names[:-1])}"
        f"{' and ' if len(explanation_names) > 1 else ''}{explanation_names[-1]}."
    )

    return RiskAssessment(
        score=final_score,
        level=level,
        breakdown={**breakdown, "final_score": final_score},
        explanation=explanation,
    )
