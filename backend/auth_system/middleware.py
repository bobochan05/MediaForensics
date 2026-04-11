from __future__ import annotations

from functools import wraps
from typing import Callable, TypeVar

from flask import g, jsonify, redirect, request, url_for

F = TypeVar("F", bound=Callable)


def extract_bearer_token(req=None) -> str | None:
    source = req or request
    raw = source.headers.get("Authorization", "")
    if not raw:
        return None
    parts = raw.strip().split(" ", 1)
    if len(parts) != 2:
        return None
    if parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def jwt_required(auth_service, *, allow_guest: bool = False, redirect_on_fail: bool = False):
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapped(*args, **kwargs):
            principal, error = auth_service.authenticate_request(request)
            if error is not None:
                if redirect_on_fail:
                    return redirect(url_for("index"))
                return (
                    jsonify({"error": error["message"], "code": error["code"]}),
                    401,
                )
            if not allow_guest and principal.get("auth_mode") != "user":
                if redirect_on_fail:
                    return redirect(url_for("index"))
                return jsonify({"error": "User authentication is required."}), 401
            g.auth_principal = principal
            return func(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorator
