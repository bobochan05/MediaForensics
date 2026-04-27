from __future__ import annotations

"""WSGI entrypoint for Gunicorn.

This module intentionally matches the common tutorial-style command:

    gunicorn your_application.wsgi

Gunicorn will import this module and look for `application` by default.
"""

try:
    # When running from repo root.
    from backend.app import app as _flask_app
except ModuleNotFoundError:
    # When Render sets Root Directory to `backend`.
    from app import app as _flask_app

app = _flask_app
application = _flask_app
