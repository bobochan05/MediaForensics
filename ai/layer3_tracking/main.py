from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai.layer3_tracking.api.routes import router
from ai.layer3_tracking.db.database import Base, engine, ensure_layer3_schema
from ai.layer3_tracking.tracker.scheduler import Layer3Scheduler
from ai.layer3_tracking.tracker.tracker import TrackingService
from ai.layer3_tracking.utils.logger import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    Base.metadata.create_all(bind=engine)
    ensure_layer3_schema(engine)

    tracking_service = TrackingService()
    scheduler = Layer3Scheduler(tracking_service)

    app.state.tracking_service = tracking_service
    app.state.scheduler = scheduler
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="Layer 3 Tracking Service", version="1.0.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
