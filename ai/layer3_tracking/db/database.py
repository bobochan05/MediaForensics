from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


load_dotenv()


class Base(DeclarativeBase):
    pass


def get_database_url() -> str:
    default_sqlite = (Path(__file__).resolve().parents[3] / "artifacts" / "layer3" / "layer3_tracking.db").resolve()
    default_sqlite.parent.mkdir(parents=True, exist_ok=True)
    return os.getenv(
        "LAYER3_DATABASE_URL",
        f"sqlite:///{default_sqlite}",
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

        if inspector.has_table("content"):
            content_columns = {column["name"] for column in inspector.get_columns("content")}
            if "perceptual_hash" not in content_columns:
                connection.execute(text("ALTER TABLE content ADD COLUMN perceptual_hash VARCHAR(64)"))
            if "media_type" not in content_columns:
                connection.execute(text("ALTER TABLE content ADD COLUMN media_type VARCHAR(32)"))
            if "detection_score" not in content_columns:
                connection.execute(text("ALTER TABLE content ADD COLUMN detection_score DOUBLE PRECISION NOT NULL DEFAULT 0"))
            if "embedding_path" not in content_columns:
                connection.execute(text("ALTER TABLE content ADD COLUMN embedding_path TEXT"))
            if "owner_user_id" not in content_columns:
                connection.execute(text("ALTER TABLE content ADD COLUMN owner_user_id INTEGER"))
            if "session_scope_id" not in content_columns:
                connection.execute(text("ALTER TABLE content ADD COLUMN session_scope_id VARCHAR(96)"))
            if "cluster_id" not in content_columns:
                cluster_type = "UUID" if dialect == "postgresql" else "CHAR(32)"
                connection.execute(text(f"ALTER TABLE content ADD COLUMN cluster_id {cluster_type}"))
            if "similar_count" not in content_columns:
                connection.execute(text("ALTER TABLE content ADD COLUMN similar_count INTEGER NOT NULL DEFAULT 0"))
            if "tracking_enabled" not in content_columns:
                enabled_type = "BOOLEAN" if dialect != "sqlite" else "INTEGER"
                enabled_default = "TRUE" if dialect != "sqlite" else "1"
                connection.execute(text(f"ALTER TABLE content ADD COLUMN tracking_enabled {enabled_type} NOT NULL DEFAULT {enabled_default}"))
            if "alert_email_enabled" not in content_columns:
                enabled_type = "BOOLEAN" if dialect != "sqlite" else "INTEGER"
                enabled_default = "FALSE" if dialect != "sqlite" else "0"
                connection.execute(text(f"ALTER TABLE content ADD COLUMN alert_email_enabled {enabled_type} NOT NULL DEFAULT {enabled_default}"))
            if "alert_frequency" not in content_columns:
                connection.execute(text("ALTER TABLE content ADD COLUMN alert_frequency VARCHAR(32)"))

        if not inspector.has_table("content_clusters"):
            Base.metadata.create_all(bind=connection, tables=[Base.metadata.tables["content_clusters"]])

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
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_content_perceptual_hash ON content (perceptual_hash)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_content_media_type ON content (media_type)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_content_owner_user_id ON content (owner_user_id)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_content_session_scope_id ON content (session_scope_id)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_content_cluster_id ON content (cluster_id)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_content_clusters_centroid_hash ON content_clusters (centroid_hash)"))
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
