"""WSGI package wrapper (backend-root compatible).

This allows Render start commands like:
  gunicorn mediaforensics.wsgi:application
when the Render Root Directory is set to `backend/`.
"""
