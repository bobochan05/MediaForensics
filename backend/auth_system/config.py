from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    cleaned = value.strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class AuthSettings:
    database_url: str
    jwt_secret: str
    jwt_refresh_secret: str
    jwt_algorithm: str
    access_token_minutes: int
    refresh_token_days: int
    bcrypt_rounds: int
    rate_limit_per_minute: int
    guest_max_tries: int
    google_client_id: str | None
    access_cookie_name: str
    refresh_cookie_name: str
    cookie_secure: bool
    cookie_samesite: str
    cookie_domain: str | None


def load_auth_settings(project_dir: Path) -> AuthSettings:
    artifacts_dir = project_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    database_url = os.getenv("AUTH_DATABASE_URL", f"sqlite:///{(artifacts_dir / 'auth.db').as_posix()}")

    environment = (os.getenv("FLASK_ENV") or os.getenv("ENVIRONMENT") or "development").strip().lower()
    is_production = environment in {"production", "prod"}

    jwt_secret = (os.getenv("JWT_SECRET") or "").strip()
    if not jwt_secret:
        if is_production:
            raise RuntimeError("JWT_SECRET must be configured in environment variables.")
        jwt_secret = secrets.token_urlsafe(48)

    jwt_refresh_secret = (os.getenv("JWT_REFRESH_SECRET") or "").strip()
    if not jwt_refresh_secret:
        if is_production:
            raise RuntimeError("JWT_REFRESH_SECRET must be configured in environment variables.")
        jwt_refresh_secret = secrets.token_urlsafe(48)
    jwt_algorithm = (os.getenv("JWT_ALGORITHM") or "HS256").strip()
    access_token_minutes = int(os.getenv("JWT_ACCESS_TOKEN_MINUTES", "15"))
    refresh_token_days = int(os.getenv("JWT_REFRESH_TOKEN_DAYS", "14"))
    bcrypt_rounds = int(os.getenv("BCRYPT_ROUNDS", "12"))
    rate_limit_per_minute = int(os.getenv("AUTH_RATE_LIMIT_PER_MINUTE", "5"))
    guest_max_tries = int(os.getenv("AUTH_GUEST_MAX_TRIES", "5"))
    google_client_id = (os.getenv("GOOGLE_OAUTH_CLIENT_ID") or "").strip() or None
    access_cookie_name = (os.getenv("AUTH_ACCESS_COOKIE_NAME") or "tracelyt_access_token").strip()
    refresh_cookie_name = (os.getenv("AUTH_REFRESH_COOKIE_NAME") or "tracelyt_refresh_token").strip()

    default_secure = (os.getenv("FLASK_ENV") or "").strip().lower() == "production"
    cookie_secure = _as_bool(os.getenv("AUTH_COOKIE_SECURE"), default_secure)
    cookie_samesite = (os.getenv("AUTH_COOKIE_SAMESITE") or "Lax").strip().capitalize()
    if cookie_samesite not in {"Lax", "Strict", "None"}:
        cookie_samesite = "Lax"
    cookie_domain = (os.getenv("AUTH_COOKIE_DOMAIN") or "").strip() or None

    return AuthSettings(
        database_url=database_url,
        jwt_secret=jwt_secret,
        jwt_refresh_secret=jwt_refresh_secret,
        jwt_algorithm=jwt_algorithm,
        access_token_minutes=access_token_minutes,
        refresh_token_days=refresh_token_days,
        bcrypt_rounds=bcrypt_rounds,
        rate_limit_per_minute=rate_limit_per_minute,
        guest_max_tries=guest_max_tries,
        google_client_id=google_client_id,
        access_cookie_name=access_cookie_name,
        refresh_cookie_name=refresh_cookie_name,
        cookie_secure=cookie_secure,
        cookie_samesite=cookie_samesite,
        cookie_domain=cookie_domain,
    )
