from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from .models import Base
from .passwords import hash_password


class AuthDatabase:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = self._create_engine(database_url)
        self._session_factory = scoped_session(
            sessionmaker(bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False)
        )

    @staticmethod
    def _create_engine(database_url: str) -> Engine:
        kwargs: dict[str, object] = {"pool_pre_ping": True, "future": True}
        if database_url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        return create_engine(database_url, **kwargs)

    def init_schema(self) -> None:
        Base.metadata.create_all(self.engine)
        self._ensure_legacy_compatibility()

    def _ensure_legacy_compatibility(self) -> None:
        if not self.database_url.startswith("sqlite"):
            return
        with self.engine.begin() as connection:
            table_exists = connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            ).fetchone()
            if not table_exists:
                return
            columns = {
                str(row[1])
                for row in connection.execute(text("PRAGMA table_info(users)")).fetchall()
            }
            if "username" not in columns:
                connection.execute(text("ALTER TABLE users ADD COLUMN username TEXT"))
            if "updated_at" not in columns:
                connection.execute(text("ALTER TABLE users ADD COLUMN updated_at TEXT"))
            if "created_at" not in columns:
                connection.execute(text("ALTER TABLE users ADD COLUMN created_at TEXT"))
            if "password_hash" not in columns:
                connection.execute(text("ALTER TABLE users ADD COLUMN password_hash TEXT"))
                if "password" in columns:
                    connection.execute(
                        text("UPDATE users SET password_hash = password WHERE password_hash IS NULL AND password IS NOT NULL")
                    )

            now_iso = datetime.now(timezone.utc).isoformat()
            connection.execute(
                text("UPDATE users SET username = 'user_' || id WHERE username IS NULL OR TRIM(username) = ''")
            )
            connection.execute(
                text(
                    "UPDATE users SET updated_at = COALESCE(updated_at, created_at) "
                    "WHERE updated_at IS NULL OR TRIM(updated_at) = ''"
                )
            )
            connection.execute(
                text("UPDATE users SET created_at = :now_iso WHERE created_at IS NULL OR TRIM(created_at) = ''"),
                {"now_iso": now_iso},
            )
            connection.execute(
                text("UPDATE users SET updated_at = :now_iso WHERE updated_at IS NULL OR TRIM(updated_at) = ''"),
                {"now_iso": now_iso},
            )

            rows = connection.execute(text("SELECT id, password_hash FROM users")).fetchall()
            for row in rows:
                user_id = int(row[0])
                stored = str(row[1] or "").strip()
                if not stored:
                    continue
                # Keep known hash formats; hash any legacy plain-text values.
                if stored.startswith("$2") or stored.startswith("pbkdf2:") or stored.startswith("scrypt:"):
                    continue
                upgraded = hash_password(stored, rounds=12)
                connection.execute(
                    text("UPDATE users SET password_hash = :password_hash WHERE id = :user_id"),
                    {"password_hash": upgraded, "user_id": user_id},
                )

            try:
                connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique ON users(email)"))
            except Exception:
                pass
            try:
                connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_unique ON users(username)"))
            except Exception:
                pass

    def session(self) -> Session:
        return self._session_factory()

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        session = self.session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def remove_scoped_session(self) -> None:
        self._session_factory.remove()
