from __future__ import annotations

import os
from collections.abc import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


load_dotenv()


class Base(DeclarativeBase):
    pass


def get_database_url() -> str:
    return os.getenv(
        "LAYER3_DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/deepfake_layer3",
    )


def create_db_engine(database_url: str | None = None) -> Engine:
    url = database_url or get_database_url()
    connect_args: dict[str, object] = {}
    engine_kwargs: dict[str, object] = {
        "future": True,
        "pool_pre_ping": True,
    }

    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    else:
        engine_kwargs.update(
            {
                "pool_size": int(os.getenv("LAYER3_DB_POOL_SIZE", "5")),
                "max_overflow": int(os.getenv("LAYER3_DB_MAX_OVERFLOW", "10")),
            }
        )

    return create_engine(url, connect_args=connect_args, **engine_kwargs)


def ensure_layer3_schema(engine: Engine) -> None:
    dialect = engine.dialect.name
    inspector = inspect(engine)

    with engine.begin() as connection:
        if dialect == "postgresql":
            connection.execute(text("ALTER TYPE content_status ADD VALUE IF NOT EXISTS 'slow_spread'"))

        if inspector.has_table("tracking_logs"):
            columns = {column["name"] for column in inspector.get_columns("tracking_logs")}
            if "growth_velocity" not in columns:
                connection.execute(text("ALTER TABLE tracking_logs ADD COLUMN growth_velocity DOUBLE PRECISION NOT NULL DEFAULT 0"))
            if "spread_score" not in columns:
                connection.execute(text("ALTER TABLE tracking_logs ADD COLUMN spread_score DOUBLE PRECISION NOT NULL DEFAULT 0"))
            if "success" not in columns:
                success_type = "BOOLEAN" if dialect != "sqlite" else "INTEGER"
                success_default = "TRUE" if dialect != "sqlite" else "1"
                connection.execute(
                    text(f"ALTER TABLE tracking_logs ADD COLUMN success {success_type} NOT NULL DEFAULT {success_default}")
                )
            if "failure_reason" not in columns:
                connection.execute(text("ALTER TABLE tracking_logs ADD COLUMN failure_reason TEXT"))

        if dialect in {"postgresql", "sqlite"}:
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_sources_content_id ON sources (content_id)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_tracking_logs_checked_at ON tracking_logs (checked_at)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_tracking_logs_content_id_checked_at ON tracking_logs (content_id, checked_at)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_api_usage_date ON api_usage (date)"))


engine = create_db_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
