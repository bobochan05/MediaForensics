import os

# Bind explicitly to Render's assigned port.
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"

# Keep default low; Render sets WEB_CONCURRENCY automatically.
workers = int(os.getenv("WEB_CONCURRENCY", "1"))

# Log to stdout/stderr so Render logs show startup failures.
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
capture_output = True

# Load Flask before binding the public socket so Render only sees an open port
# after the app is ready to answer health checks.
preload_app = True

# Avoid slow /tmp-backed worker heartbeat files on hosted Linux filesystems.
worker_tmp_dir = "/dev/shm"

# Give slower cold starts (model loading) more time after the port is bound.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
