from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

from .config import AuthSettings


class TokenValidationError(Exception):
    pass


class TokenExpiredError(TokenValidationError):
    pass


@dataclass(frozen=True)
class TokenBundle:
    access_token: str
    refresh_token: str
    access_claims: dict[str, Any]
    refresh_claims: dict[str, Any]



def _utc_now() -> datetime:
    return datetime.now(timezone.utc)



def _build_common_claims(token_type: str, expires_at: datetime) -> dict[str, Any]:
    now = _utc_now()
    return {
        "jti": uuid4().hex,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }



def issue_token_pair_for_user(user_id: int, email: str, username: str, settings: AuthSettings) -> TokenBundle:
    now = _utc_now()
    access_exp = now + timedelta(minutes=settings.access_token_minutes)
    refresh_exp = now + timedelta(days=settings.refresh_token_days)

    access_claims = _build_common_claims("access", access_exp)
    access_claims.update(
        {
            "sub": str(user_id),
            "auth_mode": "user",
            "user_id": int(user_id),
            "email": email,
            "username": username,
        }
    )

    refresh_claims = _build_common_claims("refresh", refresh_exp)
    refresh_claims.update(
        {
            "sub": str(user_id),
            "auth_mode": "user",
            "user_id": int(user_id),
            "email": email,
            "username": username,
        }
    )

    return TokenBundle(
        access_token=jwt.encode(access_claims, settings.jwt_secret, algorithm=settings.jwt_algorithm),
        refresh_token=jwt.encode(refresh_claims, settings.jwt_refresh_secret, algorithm=settings.jwt_algorithm),
        access_claims=access_claims,
        refresh_claims=refresh_claims,
    )



def issue_token_pair_for_guest(guest_id: str, settings: AuthSettings) -> TokenBundle:
    now = _utc_now()
    access_exp = now + timedelta(minutes=settings.access_token_minutes)
    refresh_exp = now + timedelta(days=settings.refresh_token_days)

    access_claims = _build_common_claims("access", access_exp)
    access_claims.update(
        {
            "sub": guest_id,
            "auth_mode": "guest",
            "guest_id": guest_id,
            "user_id": None,
            "email": "",
            "username": "guest",
        }
    )

    refresh_claims = _build_common_claims("refresh", refresh_exp)
    refresh_claims.update(
        {
            "sub": guest_id,
            "auth_mode": "guest",
            "guest_id": guest_id,
            "user_id": None,
            "email": "",
            "username": "guest",
        }
    )

    return TokenBundle(
        access_token=jwt.encode(access_claims, settings.jwt_secret, algorithm=settings.jwt_algorithm),
        refresh_token=jwt.encode(refresh_claims, settings.jwt_refresh_secret, algorithm=settings.jwt_algorithm),
        access_claims=access_claims,
        refresh_claims=refresh_claims,
    )



def decode_access_token(token: str, settings: AuthSettings) -> dict[str, Any]:
    return _decode_token(token, settings.jwt_secret, settings.jwt_algorithm, expected_type="access")



def decode_refresh_token(token: str, settings: AuthSettings) -> dict[str, Any]:
    return _decode_token(token, settings.jwt_refresh_secret, settings.jwt_algorithm, expected_type="refresh")



def _decode_token(token: str, secret: str, algorithm: str, expected_type: str) -> dict[str, Any]:
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[algorithm],
            options={"require": ["jti", "iat", "exp", "type"]},
        )
    except ExpiredSignatureError as exc:
        raise TokenExpiredError("Token expired.") from exc
    except InvalidTokenError as exc:
        raise TokenValidationError("Invalid token.") from exc

    if claims.get("type") != expected_type:
        raise TokenValidationError("Invalid token type.")

    return claims
