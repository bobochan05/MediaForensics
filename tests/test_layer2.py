from __future__ import annotations

from ai.layer2_matching.credibility import source_credibility_score


def test_layer2_credibility_score_bounds():
    score = source_credibility_score("https://www.reuters.com/world", platform="news")
    assert 0.0 <= score <= 1.0

