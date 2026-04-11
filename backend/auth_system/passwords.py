from __future__ import annotations

import re
from typing import Final

import bcrypt
from werkzeug.security import check_password_hash

EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_]{3,32}$")
LOWERCASE_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z]")
UPPERCASE_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Z]")
DIGIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"\d")
SYMBOL_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9]")


class PasswordValidationError(ValueError):
    pass


def normalize_email(value: str) -> str:
    return value.strip().lower()


def normalize_username(value: str) -> str:
    return value.strip()


def validate_email(value: str) -> str:
    email = normalize_email(value)
    if not email or not EMAIL_PATTERN.fullmatch(email):
        raise ValueError("Enter a valid email address.")
    return email


def validate_username(value: str) -> str:
    username = normalize_username(value)
    if not USERNAME_PATTERN.fullmatch(username):
        raise ValueError("Username must be 3-32 characters and use letters, numbers, or underscores only.")
    return username


def validate_password_strength(password: str) -> str:
    if len(password) < 8:
        raise PasswordValidationError("Password must be at least 8 characters long.")
    if not LOWERCASE_PATTERN.search(password):
        raise PasswordValidationError("Password must include at least one lowercase letter.")
    if not UPPERCASE_PATTERN.search(password):
        raise PasswordValidationError("Password must include at least one uppercase letter.")
    if not DIGIT_PATTERN.search(password):
        raise PasswordValidationError("Password must include at least one number.")
    if not SYMBOL_PATTERN.search(password):
        raise PasswordValidationError("Password must include at least one special character.")
    return password


def hash_password(password: str, rounds: int = 12) -> str:
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=rounds))
    return hashed.decode("utf-8")


def verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash:
        return False
    cleaned = stored_hash.strip()
    if cleaned.startswith("$2"):
        try:
            return bcrypt.checkpw(password.encode("utf-8"), cleaned.encode("utf-8"))
        except ValueError:
            return False
    return check_password_hash(cleaned, password)


def hash_needs_upgrade(stored_hash: str) -> bool:
    return not (stored_hash or "").strip().startswith("$2")
