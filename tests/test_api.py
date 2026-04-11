from __future__ import annotations

from backend.app import app


def test_health_endpoint():
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert payload.get("status") == "ok"

