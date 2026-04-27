"""WSGI entrypoint for Render/Gunicorn.

Supports commands like:
  gunicorn mediaforensics.wsgi:application --bind 0.0.0.0:$PORT

The underlying Flask app lives in `backend.app` as `app`.
"""

from __future__ import annotations

try:
    # Preferred import when repository root is on PYTHONPATH.
    from backend.app import app as application
except Exception:  # pragma: no cover
    # Fallback for unusual working directories where `backend` isn't importable.
    from app import app as application  # type: ignore
