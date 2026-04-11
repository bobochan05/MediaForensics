from __future__ import annotations

from pathlib import Path

from .routes import create_auth_blueprint
from .service import AuthService


def init_auth_system(app, project_dir: Path) -> AuthService:
    auth_service = AuthService(project_dir=project_dir)
    auth_service.configure_app_cookies(app)
    app.register_blueprint(create_auth_blueprint(auth_service))
    return auth_service
