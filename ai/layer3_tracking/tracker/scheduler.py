from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from threading import Lock

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from .tracker import TrackingService


LOGGER = logging.getLogger(__name__)


class Layer3Scheduler:
    def __init__(self, tracking_service: TrackingService) -> None:
        self.tracking_service = tracking_service
        self.scheduler = BackgroundScheduler(timezone="UTC")
        self._configured = False
        self._run_lock = Lock()
        self.last_run_started_at: datetime | None = None
        self.last_run_completed_at: datetime | None = None

    def _configure_jobs(self) -> None:
        if self._configured:
            return
        self.scheduler.add_job(
            self.run_scheduled_tracking,
            CronTrigger(hour="0,5,10,15,20", minute=0),
            id="layer3-track-all",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=300,
        )
        self.scheduler.add_job(
            self.run_startup_recovery,
            DateTrigger(run_date=datetime.now(timezone.utc) + timedelta(seconds=15)),
            id="layer3-startup-recovery",
            replace_existing=True,
            max_instances=1,
        )
        self._configured = True

    def start(self) -> None:
        self._configure_jobs()
        if not self.scheduler.running:
            LOGGER.info("Starting Layer 3 scheduler")
            self.scheduler.start()

    def shutdown(self) -> None:
        if self.scheduler.running:
            LOGGER.info("Shutting down Layer 3 scheduler")
            self.scheduler.shutdown(wait=False)

    def run_scheduled_tracking(self) -> None:
        if not self._run_lock.acquire(blocking=False):
            LOGGER.warning("Skipping scheduled Layer 3 tracking cycle because another run is active")
            return
        try:
            self.last_run_started_at = datetime.now(timezone.utc)
            LOGGER.info("Running scheduled Layer 3 tracking cycle")
            self.tracking_service.track_all_content()
            self.last_run_completed_at = datetime.now(timezone.utc)
        finally:
            self._run_lock.release()

    def run_startup_recovery(self) -> None:
        LOGGER.info("Running Layer 3 startup recovery cycle")
        self.run_scheduled_tracking()
