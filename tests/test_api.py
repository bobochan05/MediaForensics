from __future__ import annotations

from backend.app import (
    AGENT_INSUFFICIENT_EVIDENCE_REPLY,
    AGENT_OUT_OF_SCOPE_REPLY,
    _agent_context_ready,
    _agent_policy_reply,
    _build_agent_context,
    app,
    build_layer2_response,
)


def test_health_endpoint():
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert payload.get("status") == "ok"


def test_build_layer2_response_includes_dashboard_matches():
    payload = build_layer2_response(
        {
            "exact_matches": [
                {
                    "id": "exact-1",
                    "url": "https://example.com/story",
                    "media_url": "https://cdn.example.com/image.jpg",
                    "title": "Example story",
                    "platform": "news",
                    "timestamp": "2026-04-12T00:00:00+00:00",
                    "visual_similarity": 0.97,
                    "metadata": {"final_score": 0.96},
                }
            ],
            "visual_matches_top10": [],
            "embedding_matches_top10": [],
        }
    )

    assert payload["counts"] == {"exact": 1, "visual": 0, "embedding": 0}
    assert payload["count"] == 1
    assert len(payload["matches"]) == 1
    assert payload["matches"][0]["source_url"] == "https://example.com/story"
    assert payload["matches"][0]["preview_url"] == "https://cdn.example.com/image.jpg"
    assert payload["matches"][0]["similarity"] > 0.9


def test_agent_context_ready_requires_real_analysis_signal():
    empty_context = _build_agent_context({}, {}, {})
    assert _agent_context_ready(empty_context) is False

    populated_context = _build_agent_context({"result": "FAKE", "confidence": 0.91}, {}, {})
    assert _agent_context_ready(populated_context) is True


def test_agent_policy_reply_rejects_out_of_scope_questions():
    context = _build_agent_context({"result": "FAKE", "confidence": 0.91}, {}, {})
    reply = _agent_policy_reply("Who is the person in this video?", context)
    assert reply == AGENT_OUT_OF_SCOPE_REPLY


def test_agent_policy_reply_requires_evidence_for_in_scope_questions():
    context = _build_agent_context({}, {}, {})
    reply = _agent_policy_reply("Is this media authentic?", context)
    assert reply == AGENT_INSUFFICIENT_EVIDENCE_REPLY

