"""WSGI entrypoint for Render/Gunicorn (backend-root compatible).

Supports commands like:
  gunicorn mediaforensics.wsgi:application --bind 0.0.0.0:$PORT

When running from the `backend/` directory, the Flask app module is `app.py`.
"""

from __future__ import annotations

from app import app as application
