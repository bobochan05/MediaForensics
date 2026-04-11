from __future__ import annotations

from flask import Flask


def register_auth_routes(app: Flask, auth_service) -> None:
    """Attach authentication routes exposed by the auth service blueprint."""
    if getattr(auth_service, "blueprint", None) is not None:
        app.register_blueprint(auth_service.blueprint)

