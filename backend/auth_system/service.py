from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from flask import Request
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import AuthSettings, load_auth_settings
from .db import AuthDatabase
from .jwt_utils import (
    TokenBundle,
    TokenExpiredError,
    TokenValidationError,
    decode_access_token,
    decode_refresh_token,
    issue_token_pair_for_guest,
    issue_token_pair_for_user,
)
from .middleware import extract_bearer_token
from .models import AuditLog, GuestUsage, RefreshToken, RevokedToken, User, utcnow
from .passwords import (
    PasswordValidationError,
    hash_needs_upgrade,
    hash_password,
    normalize_username,
    validate_email,
    validate_password_strength,
    validate_username,
    verify_password,
)
from .rate_limit import InMemoryRateLimiter

try:
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token as google_id_token
except Exception:  # pragma: no cover - optional dependency at runtime
    google_requests = None
    google_id_token = None


@dataclass(frozen=True)
class ServiceResult:
    payload: dict[str, Any]
    status_code: int
    token_bundle: TokenBundle | None = None


class AuthService:
    def __init__(self, project_dir: Path):
        self.settings: AuthSettings = load_auth_settings(project_dir)
        self.db = AuthDatabase(self.settings.database_url)
        self.db.init_schema()
        self.limiter = InMemoryRateLimiter(self.settings.rate_limit_per_minute, window_seconds=60)

    def configure_app_cookies(self, app) -> None:
        app.config["SESSION_COOKIE_HTTPONLY"] = True
        app.config["SESSION_COOKIE_SECURE"] = self.settings.cookie_secure
        app.config["SESSION_COOKIE_SAMESITE"] = self.settings.cookie_samesite

    def set_auth_cookies(self, response, token_bundle: TokenBundle) -> None:
        response.set_cookie(
            self.settings.access_cookie_name,
            token_bundle.access_token,
            max_age=self.settings.access_token_minutes * 60,
            httponly=True,
            secure=self.settings.cookie_secure,
            samesite=self.settings.cookie_samesite,
            path="/",
            domain=self.settings.cookie_domain,
        )
        response.set_cookie(
            self.settings.refresh_cookie_name,
            token_bundle.refresh_token,
            max_age=self.settings.refresh_token_days * 24 * 60 * 60,
            httponly=True,
            secure=self.settings.cookie_secure,
            samesite=self.settings.cookie_samesite,
            path="/api/auth",
            domain=self.settings.cookie_domain,
        )

    def clear_auth_cookies(self, response) -> None:
        response.set_cookie(
            self.settings.access_cookie_name,
            "",
            expires=0,
            httponly=True,
            secure=self.settings.cookie_secure,
            samesite=self.settings.cookie_samesite,
            path="/",
            domain=self.settings.cookie_domain,
        )
        response.set_cookie(
            self.settings.refresh_cookie_name,
            "",
            expires=0,
            httponly=True,
            secure=self.settings.cookie_secure,
            samesite=self.settings.cookie_samesite,
            path="/api/auth",
            domain=self.settings.cookie_domain,
        )

    def _check_rate_limit(self, action: str, req: Request, identity: str | None = None) -> tuple[bool, int]:
        ip = self._request_ip(req)
        identity_key = (identity or "").strip().lower()
        scoped_keys = [f"{action}:{ip}"]
        if identity_key:
            scoped_keys.append(f"{action}:{ip}:{identity_key}")

        retry_after = 0
        for key in scoped_keys:
            allowed, retry = self.limiter.allow(key)
            if not allowed:
                retry_after = max(retry_after, retry)
                return False, retry_after
        return True, 0

    @staticmethod
    def _request_ip(req: Request) -> str:
        forwarded = str(req.headers.get("X-Forwarded-For") or "").strip()
        if forwarded:
            return forwarded.split(",")[0].strip()[:64]
        return str(req.remote_addr or "")[:64]

    @staticmethod
    def _request_user_agent(req: Request) -> str:
        return str(req.headers.get("User-Agent") or "")[:255]

    @staticmethod
    def _slug_username(value: str) -> str:
        lowered = value.strip().lower()
        cleaned = re.sub(r"[^a-z0-9_]+", "_", lowered)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        if not cleaned:
            return "user"
        return normalize_username(cleaned[:32])

    def _unique_username(self, session: Session, base_username: str) -> str:
        seed = base_username or "user"
        if not re.fullmatch(r"[A-Za-z0-9_]{3,32}", seed):
            seed = "user"
        candidate = seed
        counter = 1
        while session.query(User.id).filter(User.username == candidate).first() is not None:
            suffix = f"_{counter}"
            trimmed_seed = seed[: max(3, 32 - len(suffix))]
            candidate = f"{trimmed_seed}{suffix}"
            counter += 1
        return candidate

    def signup(self, payload: dict[str, Any], req: Request) -> ServiceResult:
        allowed, retry_after = self._check_rate_limit("signup", req)
        if not allowed:
            return ServiceResult(
                {
                    "error": "Too many signup attempts. Please wait and try again.",
                    "retry_after": retry_after,
                },
                429,
            )

        username_raw = str(payload.get("username") or "")
        email_raw = str(payload.get("email") or "")
        password = str(payload.get("password") or "")
        confirm_password = str(payload.get("confirm_password") or "")

        try:
            username = validate_username(username_raw)
            email = validate_email(email_raw)
            validate_password_strength(password)
        except (ValueError, PasswordValidationError) as exc:
            return ServiceResult({"error": str(exc)}, 400)

        if password != confirm_password:
            return ServiceResult({"error": "Password and confirm password do not match."}, 400)

        token_bundle: TokenBundle | None = None
        created_user: User | None = None
        try:
            with self.db.session_scope() as session:
                existing_email = session.query(User).filter(User.email == email).one_or_none()
                if existing_email is not None:
                    self._write_audit(
                        session,
                        event="signup",
                        success=False,
                        req=req,
                        email=email,
                        detail="duplicate_email",
                    )
                    return ServiceResult({"error": "An account with this email already exists."}, 409)

                existing_username = session.query(User).filter(User.username == username).one_or_none()
                if existing_username is not None:
                    self._write_audit(
                        session,
                        event="signup",
                        success=False,
                        req=req,
                        email=email,
                        detail="duplicate_username",
                    )
                    return ServiceResult({"error": "This username is already taken."}, 409)

                now = utcnow()
                created_user = User(
                    username=username,
                    email=email,
                    password_hash=hash_password(password, rounds=self.settings.bcrypt_rounds),
                    created_at=now,
                    updated_at=now,
                )
                session.add(created_user)
                session.flush()

                token_bundle = issue_token_pair_for_user(
                    user_id=int(created_user.id),
                    email=created_user.email,
                    username=created_user.username,
                    settings=self.settings,
                )
                self._store_refresh_token(
                    session=session,
                    token_claims=token_bundle.refresh_claims,
                    auth_mode="user",
                    user_id=int(created_user.id),
                    guest_id=None,
                    req=req,
                )
                self._write_audit(
                    session,
                    event="signup",
                    success=True,
                    req=req,
                    email=created_user.email,
                    user_id=int(created_user.id),
                    auth_mode="user",
                    detail="signup_success",
                )
        except IntegrityError:
            return ServiceResult({"error": "Could not create account. Please try again."}, 409)
        except Exception:
            return ServiceResult({"error": "Server error while creating account."}, 500)

        assert token_bundle is not None and created_user is not None
        response_payload = self._build_auth_success_payload(
            auth_state="user",
            message="Account created successfully.",
            user=created_user,
            guest_usage=self._guest_usage_payload(None),
            token_bundle=token_bundle,
        )
        return ServiceResult(response_payload, 201, token_bundle)

    def login(self, payload: dict[str, Any], req: Request) -> ServiceResult:
        identifier_raw = str(
            payload.get("email")
            or payload.get("identifier")
            or payload.get("username")
            or ""
        ).strip()
        password = str(payload.get("password") or "")

        if not identifier_raw or not password:
            return ServiceResult({"error": "Email/username and password are required."}, 400)

        identifier_type = "email"
        normalized_username = ""
        identity_for_limit = identifier_raw.lower()
        if "@" in identifier_raw:
            try:
                identity_for_limit = validate_email(identifier_raw)
            except ValueError as exc:
                return ServiceResult({"error": str(exc)}, 400)
        else:
            identifier_type = "username"
            try:
                normalized_username = validate_username(identifier_raw)
            except ValueError as exc:
                return ServiceResult({"error": str(exc)}, 400)
            identity_for_limit = normalized_username.lower()

        allowed, retry_after = self._check_rate_limit("login", req, identity=identity_for_limit)
        if not allowed:
            return ServiceResult(
                {
                    "error": "Too many login attempts. Please wait and try again.",
                    "retry_after": retry_after,
                },
                429,
            )

        token_bundle: TokenBundle | None = None
        logged_user: User | None = None
        with self.db.session_scope() as session:
            if identifier_type == "email":
                user = session.query(User).filter(User.email == identity_for_limit).one_or_none()
            else:
                user = session.query(User).filter(User.username == normalized_username).one_or_none()
            if user is None or not verify_password(password, user.password_hash):
                self._write_audit(
                    session,
                    event="login",
                    success=False,
                    req=req,
                    email=(identity_for_limit if identifier_type == "email" else None),
                    detail="invalid_credentials",
                )
                return ServiceResult({"error": "Invalid email or password."}, 401)

            if hash_needs_upgrade(user.password_hash):
                user.password_hash = hash_password(password, rounds=self.settings.bcrypt_rounds)

            user.updated_at = utcnow()
            token_bundle = issue_token_pair_for_user(
                user_id=int(user.id),
                email=user.email,
                username=user.username,
                settings=self.settings,
            )
            self._store_refresh_token(
                session=session,
                token_claims=token_bundle.refresh_claims,
                auth_mode="user",
                user_id=int(user.id),
                guest_id=None,
                req=req,
            )
            self._write_audit(
                session,
                event="login",
                success=True,
                req=req,
                email=user.email,
                user_id=int(user.id),
                auth_mode="user",
                detail="login_success",
            )
            logged_user = user

        assert token_bundle is not None and logged_user is not None
        payload_out = self._build_auth_success_payload(
            auth_state="user",
            message="Login successful.",
            user=logged_user,
            guest_usage=self._guest_usage_payload(None),
            token_bundle=token_bundle,
        )
        return ServiceResult(payload_out, 200, token_bundle)

    def google_login(self, payload: dict[str, Any], req: Request) -> ServiceResult:
        allowed, retry_after = self._check_rate_limit("google", req)
        if not allowed:
            return ServiceResult(
                {"error": "Too many authentication attempts. Please wait and try again.", "retry_after": retry_after},
                429,
            )

        if google_id_token is None or google_requests is None:
            return ServiceResult({"error": "Google OAuth dependency is not installed on the server."}, 500)

        if not self.settings.google_client_id:
            return ServiceResult({"error": "Google OAuth is not configured for this server."}, 503)

        raw_token = str(payload.get("id_token") or "")
        if not raw_token:
            return ServiceResult({"error": "Google ID token is required."}, 400)

        try:
            token_info = google_id_token.verify_oauth2_token(
                raw_token,
                google_requests.Request(),
                self.settings.google_client_id,
            )
        except Exception:
            return ServiceResult({"error": "Google authentication failed."}, 401)

        if not token_info.get("email_verified"):
            return ServiceResult({"error": "Google account email is not verified."}, 401)

        try:
            email = validate_email(str(token_info.get("email") or ""))
        except ValueError as exc:
            return ServiceResult({"error": str(exc)}, 400)

        name = str(token_info.get("name") or "").strip()
        fallback_username = email.split("@", 1)[0]
        preferred_username = self._slug_username(name or fallback_username)

        token_bundle: TokenBundle | None = None
        oauth_user: User | None = None
        with self.db.session_scope() as session:
            user = session.query(User).filter(User.email == email).one_or_none()
            if user is None:
                username = self._unique_username(session, preferred_username)
                now = utcnow()
                user = User(
                    username=username,
                    email=email,
                    password_hash=hash_password(secrets.token_urlsafe(32), rounds=self.settings.bcrypt_rounds),
                    created_at=now,
                    updated_at=now,
                )
                session.add(user)
                session.flush()
            token_bundle = issue_token_pair_for_user(
                user_id=int(user.id),
                email=user.email,
                username=user.username,
                settings=self.settings,
            )
            self._store_refresh_token(
                session=session,
                token_claims=token_bundle.refresh_claims,
                auth_mode="user",
                user_id=int(user.id),
                guest_id=None,
                req=req,
            )
            self._write_audit(
                session,
                event="google_login",
                success=True,
                req=req,
                email=user.email,
                user_id=int(user.id),
                auth_mode="user",
                detail="google_oauth_success",
            )
            oauth_user = user

        assert token_bundle is not None and oauth_user is not None
        payload_out = self._build_auth_success_payload(
            auth_state="user",
            message="Google login successful.",
            user=oauth_user,
            guest_usage=self._guest_usage_payload(None),
            token_bundle=token_bundle,
        )
        return ServiceResult(payload_out, 200, token_bundle)

    def guest_login(self, req: Request) -> ServiceResult:
        allowed, retry_after = self._check_rate_limit("guest", req)
        if not allowed:
            return ServiceResult(
                {"error": "Too many guest login attempts. Please wait and try again.", "retry_after": retry_after},
                429,
            )

        guest_id = uuid4().hex
        token_bundle: TokenBundle | None = None
        with self.db.session_scope() as session:
            now = utcnow()
            usage = GuestUsage(
                guest_id=guest_id,
                used_count=0,
                limit_count=self.settings.guest_max_tries,
                created_at=now,
                updated_at=now,
            )
            session.add(usage)
            token_bundle = issue_token_pair_for_guest(guest_id=guest_id, settings=self.settings)
            self._store_refresh_token(
                session=session,
                token_claims=token_bundle.refresh_claims,
                auth_mode="guest",
                user_id=None,
                guest_id=guest_id,
                req=req,
            )
            self._write_audit(
                session,
                event="guest_login",
                success=True,
                req=req,
                auth_mode="guest",
                detail="guest_session_started",
            )

        assert token_bundle is not None
        usage_payload = self._guest_usage_payload(guest_id)
        payload_out = self._build_auth_success_payload(
            auth_state="guest",
            message="Guest mode activated.",
            user=None,
            guest_usage=usage_payload,
            token_bundle=token_bundle,
        )
        payload_out["guest_id"] = guest_id
        return ServiceResult(payload_out, 200, token_bundle)

    def refresh(self, req: Request) -> ServiceResult:
        raw_refresh_token = self._extract_token(req, token_kind="refresh")
        if not raw_refresh_token:
            return ServiceResult({"error": "Refresh token is required.", "code": "auth_required"}, 401)

        try:
            claims = decode_refresh_token(raw_refresh_token, self.settings)
        except TokenExpiredError:
            return ServiceResult({"error": "Refresh token expired.", "code": "token_expired"}, 401)
        except TokenValidationError:
            return ServiceResult({"error": "Invalid refresh token.", "code": "invalid_token"}, 401)

        new_bundle: TokenBundle | None = None
        refreshed_user: User | None = None
        refreshed_guest_id: str | None = None
        with self.db.session_scope() as session:
            token_jti = str(claims.get("jti") or "")
            stored_refresh = session.query(RefreshToken).filter(RefreshToken.token_jti == token_jti).one_or_none()
            if stored_refresh is None or stored_refresh.revoked_at is not None:
                return ServiceResult({"error": "Refresh token is not valid.", "code": "invalid_token"}, 401)

            if self._is_token_revoked(session, token_jti):
                return ServiceResult({"error": "Refresh token is revoked.", "code": "invalid_token"}, 401)

            auth_mode = str(claims.get("auth_mode") or "").strip().lower()
            stored_refresh.revoked_at = utcnow()

            if auth_mode == "user":
                user_id = int(claims.get("user_id") or 0)
                user = session.query(User).filter(User.id == user_id).one_or_none()
                if user is None:
                    return ServiceResult({"error": "User no longer exists.", "code": "invalid_token"}, 401)
                new_bundle = issue_token_pair_for_user(
                    user_id=int(user.id),
                    email=user.email,
                    username=user.username,
                    settings=self.settings,
                )
                refreshed_user = user
                self._store_refresh_token(
                    session=session,
                    token_claims=new_bundle.refresh_claims,
                    auth_mode="user",
                    user_id=int(user.id),
                    guest_id=None,
                    req=req,
                )
            elif auth_mode == "guest":
                guest_id = str(claims.get("guest_id") or claims.get("sub") or "").strip()
                if not guest_id:
                    return ServiceResult({"error": "Invalid guest token.", "code": "invalid_token"}, 401)
                self._ensure_guest_usage_row(session, guest_id)
                new_bundle = issue_token_pair_for_guest(guest_id=guest_id, settings=self.settings)
                refreshed_guest_id = guest_id
                self._store_refresh_token(
                    session=session,
                    token_claims=new_bundle.refresh_claims,
                    auth_mode="guest",
                    user_id=None,
                    guest_id=guest_id,
                    req=req,
                )
            else:
                return ServiceResult({"error": "Invalid token mode.", "code": "invalid_token"}, 401)

            assert new_bundle is not None
            stored_refresh.replaced_by_jti = str(new_bundle.refresh_claims.get("jti") or "")
            self._revoke_jti(
                session=session,
                jti=token_jti,
                token_type="refresh",
                reason="rotated",
                exp_timestamp=int(claims.get("exp") or 0),
            )

        assert new_bundle is not None
        if refreshed_user is not None:
            usage_payload = self._guest_usage_payload(None)
            payload_out = self._build_auth_success_payload(
                auth_state="user",
                message="Token refreshed.",
                user=refreshed_user,
                guest_usage=usage_payload,
                token_bundle=new_bundle,
            )
        else:
            usage_payload = self._guest_usage_payload(refreshed_guest_id)
            payload_out = self._build_auth_success_payload(
                auth_state="guest",
                message="Token refreshed.",
                user=None,
                guest_usage=usage_payload,
                token_bundle=new_bundle,
            )
        return ServiceResult(payload_out, 200, new_bundle)

    def logout(self, req: Request) -> ServiceResult:
        access_token = self._extract_token(req, token_kind="access")
        refresh_token = self._extract_token(req, token_kind="refresh")

        with self.db.session_scope() as session:
            if access_token:
                try:
                    access_claims = decode_access_token(access_token, self.settings)
                    self._revoke_jti(
                        session=session,
                        jti=str(access_claims.get("jti") or ""),
                        token_type="access",
                        reason="logout",
                        exp_timestamp=int(access_claims.get("exp") or 0),
                    )
                except (TokenValidationError, TokenExpiredError):
                    pass

            if refresh_token:
                try:
                    refresh_claims = decode_refresh_token(refresh_token, self.settings)
                    refresh_jti = str(refresh_claims.get("jti") or "")
                    refresh_row = session.query(RefreshToken).filter(RefreshToken.token_jti == refresh_jti).one_or_none()
                    if refresh_row is not None:
                        refresh_row.revoked_at = utcnow()
                    self._revoke_jti(
                        session=session,
                        jti=refresh_jti,
                        token_type="refresh",
                        reason="logout",
                        exp_timestamp=int(refresh_claims.get("exp") or 0),
                    )
                except (TokenValidationError, TokenExpiredError):
                    pass

            self._write_audit(
                session=session,
                event="logout",
                success=True,
                req=req,
                detail="logout_success",
            )

        return ServiceResult({"status": "ok", "message": "Logged out.", "redirect_url": "/"}, 200)

    def authenticate_request(self, req: Request) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
        raw_access_token = self._extract_token(req, token_kind="access")
        if not raw_access_token:
            return None, {"code": "auth_required", "message": "Login, sign up, or continue as guest to access this service."}

        try:
            claims = decode_access_token(raw_access_token, self.settings)
        except TokenExpiredError:
            return None, {"code": "token_expired", "message": "Access token expired."}
        except TokenValidationError:
            return None, {"code": "invalid_token", "message": "Invalid access token."}

        token_jti = str(claims.get("jti") or "")
        with self.db.session_scope() as session:
            if self._is_token_revoked(session, token_jti):
                return None, {"code": "invalid_token", "message": "Token has been revoked."}

            auth_mode = str(claims.get("auth_mode") or "").strip().lower()
            if auth_mode == "user":
                user_id = int(claims.get("user_id") or 0)
                user = session.query(User).filter(User.id == user_id).one_or_none()
                if user is None:
                    return None, {"code": "invalid_token", "message": "User account no longer exists."}
                return {
                    "auth_mode": "user",
                    "user_id": int(user.id),
                    "email": user.email,
                    "username": user.username,
                    "claims": claims,
                }, None

            if auth_mode == "guest":
                guest_id = str(claims.get("guest_id") or claims.get("sub") or "").strip()
                if not guest_id:
                    return None, {"code": "invalid_token", "message": "Guest token is invalid."}
                self._ensure_guest_usage_row(session, guest_id)
                return {
                    "auth_mode": "guest",
                    "guest_id": guest_id,
                    "user_id": None,
                    "email": "",
                    "username": "guest",
                    "claims": claims,
                }, None

        return None, {"code": "invalid_token", "message": "Unsupported authentication mode."}

    def build_session_payload(self, req: Request) -> dict[str, Any]:
        principal, _error = self.authenticate_request(req)
        if principal is None:
            return {
                "status": "ok",
                "auth_state": "anonymous",
                "has_service_access": False,
                "redirect_url": "/dashboard",
                **self._guest_usage_payload(None),
            }

        if principal["auth_mode"] == "user":
            return {
                "status": "ok",
                "auth_state": "user",
                "has_service_access": True,
                "redirect_url": "/dashboard",
                "user_email": principal.get("email") or "",
                "user_username": principal.get("username") or "",
                **self._guest_usage_payload(None),
            }

        guest_usage = self._guest_usage_payload(str(principal.get("guest_id") or ""))
        return {
            "status": "ok",
            "auth_state": "guest",
            "has_service_access": True,
            "redirect_url": "/dashboard",
            **guest_usage,
        }

    def build_template_context(self, req: Request) -> dict[str, Any]:
        payload = self.build_session_payload(req)
        return {
            "auth_state": payload.get("auth_state", "anonymous"),
            "has_service_access": bool(payload.get("has_service_access", False)),
            "user_email": str(payload.get("user_email") or ""),
            "user_username": str(payload.get("user_username") or ""),
            "guest_limit": int(payload.get("guest_limit") or self.settings.guest_max_tries),
            "guest_used": int(payload.get("guest_used") or 0),
            "guest_remaining": int(payload.get("guest_remaining") or self.settings.guest_max_tries),
        }

    def enforce_guest_quota(self, principal: dict[str, Any]) -> tuple[bool, str | None, dict[str, int]]:
        if principal.get("auth_mode") != "guest":
            usage = self._guest_usage_payload(None)
            return True, None, usage

        guest_id = str(principal.get("guest_id") or "")
        usage = self._guest_usage_payload(guest_id)
        if usage["guest_used"] >= usage["guest_limit"]:
            return (
                False,
                f"Guest mode allows up to {usage['guest_limit']} analysis tries. Please sign up to continue.",
                usage,
            )
        return True, None, usage

    def mark_guest_try_used(self, principal: dict[str, Any]) -> dict[str, int]:
        if principal.get("auth_mode") != "guest":
            return self._guest_usage_payload(None)

        guest_id = str(principal.get("guest_id") or "")
        with self.db.session_scope() as session:
            usage_row = self._ensure_guest_usage_row(session, guest_id)
            usage_row.used_count = min(usage_row.limit_count, usage_row.used_count + 1)
            usage_row.updated_at = utcnow()
        return self._guest_usage_payload(guest_id)

    def guest_usage_payload(self, guest_id: str | None) -> dict[str, int]:
        return self._guest_usage_payload(guest_id)

    def _build_auth_success_payload(
        self,
        *,
        auth_state: str,
        message: str,
        user: User | None,
        guest_usage: dict[str, int],
        token_bundle: TokenBundle,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "ok",
            "message": message,
            "auth_state": auth_state,
            "redirect_url": "/dashboard",
            "token_type": "Bearer",
            "access_token": token_bundle.access_token,
            "expires_in": self.settings.access_token_minutes * 60,
            **guest_usage,
        }
        if user is not None:
            payload["user_id"] = int(user.id)
            payload["user_email"] = user.email
            payload["user_username"] = user.username
        return payload

    def _store_refresh_token(
        self,
        *,
        session: Session,
        token_claims: dict[str, Any],
        auth_mode: str,
        user_id: int | None,
        guest_id: str | None,
        req: Request,
    ) -> None:
        exp_timestamp = int(token_claims.get("exp") or 0)
        expires_at = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
        refresh_row = RefreshToken(
            token_jti=str(token_claims.get("jti") or ""),
            user_id=user_id,
            guest_id=guest_id,
            auth_mode=auth_mode,
            issued_at=utcnow(),
            expires_at=expires_at,
            ip_address=self._request_ip(req),
            user_agent=self._request_user_agent(req),
        )
        session.add(refresh_row)

    def _is_token_revoked(self, session: Session, jti: str) -> bool:
        if not jti:
            return True
        row = session.query(RevokedToken).filter(RevokedToken.token_jti == jti).one_or_none()
        return row is not None

    def _revoke_jti(self, session: Session, *, jti: str, token_type: str, reason: str, exp_timestamp: int) -> None:
        if not jti or exp_timestamp <= 0:
            return
        existing = session.query(RevokedToken).filter(RevokedToken.token_jti == jti).one_or_none()
        if existing is not None:
            return
        revoked = RevokedToken(
            token_jti=jti,
            token_type=token_type,
            reason=reason,
            expires_at=datetime.fromtimestamp(exp_timestamp, tz=timezone.utc),
            revoked_at=utcnow(),
        )
        session.add(revoked)

    def _extract_token(self, req: Request, *, token_kind: str) -> str | None:
        bearer_token = extract_bearer_token(req)
        if bearer_token:
            return bearer_token
        if token_kind == "refresh":
            return str(req.cookies.get(self.settings.refresh_cookie_name) or "").strip() or None
        return str(req.cookies.get(self.settings.access_cookie_name) or "").strip() or None

    def _guest_usage_payload(self, guest_id: str | None) -> dict[str, int]:
        if not guest_id:
            limit = self.settings.guest_max_tries
            return {"guest_limit": limit, "guest_used": 0, "guest_remaining": limit}

        with self.db.session_scope() as session:
            usage = self._ensure_guest_usage_row(session, guest_id)
            used = max(0, int(usage.used_count))
            limit = max(1, int(usage.limit_count))
            remaining = max(0, limit - used)
            return {"guest_limit": limit, "guest_used": used, "guest_remaining": remaining}

    def _ensure_guest_usage_row(self, session: Session, guest_id: str) -> GuestUsage:
        usage = session.query(GuestUsage).filter(GuestUsage.guest_id == guest_id).one_or_none()
        if usage is not None:
            return usage
        now = utcnow()
        usage = GuestUsage(
            guest_id=guest_id,
            used_count=0,
            limit_count=self.settings.guest_max_tries,
            created_at=now,
            updated_at=now,
        )
        session.add(usage)
        session.flush()
        return usage

    def _write_audit(
        self,
        session: Session,
        *,
        event: str,
        success: bool,
        req: Request,
        email: str | None = None,
        user_id: int | None = None,
        auth_mode: str | None = None,
        detail: str | None = None,
    ) -> None:
        audit = AuditLog(
            event=event,
            success=bool(success),
            auth_mode=auth_mode,
            email=(email or "")[:320] or None,
            user_id=user_id,
            ip_address=self._request_ip(req),
            user_agent=self._request_user_agent(req),
            detail=detail,
            created_at=utcnow(),
        )
        session.add(audit)
