from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BackendRuntimeConfig:
    project_root: Path
    artifacts_dir: Path
    dashboard_frontend_url: str
    cors_allowed_origins: set[str]
    inference_timeout_seconds: int
    max_upload_size_bytes: int


def load_backend_config(project_root: str | Path) -> BackendRuntimeConfig:
    root = Path(project_root).resolve()
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cors_allowed = {
        origin.strip()
        for origin in str(os.getenv("CORS_ALLOWED_ORIGINS") or "").split(",")
        if origin.strip()
    }
    return BackendRuntimeConfig(
        project_root=root,
        artifacts_dir=artifacts_dir,
        dashboard_frontend_url=str(os.getenv("DASHBOARD_FRONTEND_URL") or "").strip(),
        cors_allowed_origins=cors_allowed,
        inference_timeout_seconds=int(os.getenv("INFERENCE_TIMEOUT_SECONDS", "180")),
        max_upload_size_bytes=int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")) * 1024 * 1024,
    )

