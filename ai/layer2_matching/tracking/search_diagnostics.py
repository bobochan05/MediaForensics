"""Reverse-search diagnostics: structured logging, input fingerprinting,
rate tracking, response quality assessment, and result consistency checking.

This module is the instrumentation backbone for making reverse search
deterministic and debuggable.  Every public helper is thread-safe.
"""
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

from PIL import Image

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_AUDIT_LOG_SIZE = 200
_BURST_WINDOW_SECONDS = 10.0
_BURST_THRESHOLD = 6  # >N requests in window → warning
_RATE_WINDOW_SECONDS = 60.0
_CONSISTENCY_HISTORY_LIMIT = 10
_CONSISTENCY_STABLE_THRESHOLD = 0.60
_CONSISTENCY_UNSTABLE_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# 1. Input Fingerprinting
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchInputFingerprint:
    """Immutable identity snapshot for an input image."""

    file_hash_sha256: str
    file_size_bytes: int
    width: int
    height: int
    mime_type: str | None
    format_name: str | None
    cloudinary_url: str | None
    transformations_applied: bool
    correlation_id: str

    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
        *,
        cloudinary_url: str | None = None,
        transformations_applied: bool = False,
        correlation_id: str = "",
    ) -> "SearchInputFingerprint":
        """Build a fingerprint by reading the file from disk."""
        path = Path(file_path)
        data = path.read_bytes()
        file_hash = hashlib.sha256(data).hexdigest()
        file_size = len(data)

        width, height, format_name = 0, 0, None
        try:
            with Image.open(path) as img:
                width, height = img.size
                format_name = img.format
        except Exception:
            pass

        mime_type, _ = mimetypes.guess_type(str(path))

        return cls(
            file_hash_sha256=file_hash,
            file_size_bytes=file_size,
            width=width,
            height=height,
            mime_type=mime_type,
            format_name=format_name,
            cloudinary_url=cloudinary_url,
            transformations_applied=transformations_applied,
            correlation_id=correlation_id or file_hash[:16],
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        filename: str = "unknown",
        *,
        cloudinary_url: str | None = None,
        transformations_applied: bool = False,
        correlation_id: str = "",
    ) -> "SearchInputFingerprint":
        """Build a fingerprint from raw bytes (e.g. a Flask upload stream)."""
        file_hash = hashlib.sha256(data).hexdigest()

        width, height, format_name = 0, 0, None
        try:
            from io import BytesIO

            with Image.open(BytesIO(data)) as img:
                width, height = img.size
                format_name = img.format
        except Exception:
            pass

        mime_type, _ = mimetypes.guess_type(filename)

        return cls(
            file_hash_sha256=file_hash,
            file_size_bytes=len(data),
            width=width,
            height=height,
            mime_type=mime_type,
            format_name=format_name,
            cloudinary_url=cloudinary_url,
            transformations_applied=transformations_applied,
            correlation_id=correlation_id or file_hash[:16],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 2. Request / Response Log Entries
# ---------------------------------------------------------------------------
@dataclass
class SearchRequestLog:
    """Snapshot of a single SerpAPI request."""

    correlation_id: str
    engine: str
    image_url: str
    params: dict[str, Any]
    timestamp_utc: float
    attempt_number: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResponseLog:
    """Snapshot of a single SerpAPI response."""

    correlation_id: str
    engine: str
    http_status: int
    result_count: int
    top_3_domains: list[str]
    top_3_scores: list[float]
    quality_status: str  # ok | partial | rate_limited | failed | empty
    latency_ms: float
    raw_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 3. Quality Gate — decides if a result is cache-worthy
# ---------------------------------------------------------------------------
def search_quality_gate(
    response: dict[str, Any],
    *,
    primary_status: str = "ok",
    fallback_status: str = "ok",
) -> tuple[bool, str, int]:
    """Evaluate whether a reverse-search result is high enough quality to cache.

    Returns:
        (should_cache, reason, recommended_ttl_seconds)
    """
    combined = response.get("reverse_search") or response
    top_matches = combined.get("top_matches") or []
    similar_images = combined.get("similar_images") or []
    errors = combined.get("errors") or []
    total_matches = len(top_matches) + len(similar_images)

    # Both providers failed → never cache
    if primary_status in ("rate_limited", "failed") and fallback_status in ("rate_limited", "failed"):
        return False, "both_providers_failed", 0

    # Rate limited on at least one side → don't cache
    if primary_status == "rate_limited" or fallback_status == "rate_limited":
        return False, "rate_limited", 0

    # Zero matches AND errors present → don't cache (could be transient failure)
    if total_matches == 0 and errors:
        return False, "empty_with_errors", 0

    # Zero matches, no errors → genuine empty result, cache with reduced TTL
    if total_matches == 0 and not errors:
        return True, "genuine_empty", 3600  # 1 hour

    # Only one engine succeeded → partial result, reduced TTL
    if primary_status != "ok" or fallback_status != "ok":
        return True, "partial_result", 3600  # 1 hour

    # Full quality result
    return True, "quality_result", 12 * 3600  # 12 hours


# ---------------------------------------------------------------------------
# 4. Rate Tracker (token-bucket style)
# ---------------------------------------------------------------------------
class _TokenBucket:
    """Simple thread-safe token-bucket rate limiter."""

    def __init__(self, rate_per_second: float, burst: int) -> None:
        self._rate = float(rate_per_second)
        self._burst = int(burst)
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = Lock()

    def acquire(self, timeout_seconds: float = 10.0) -> bool:
        """Block until a token is available or *timeout_seconds* elapses.

        Returns True if a token was acquired, False on timeout.
        """
        deadline = time.monotonic() + timeout_seconds
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)


# ---------------------------------------------------------------------------
# 5. Result Consistency Tracker
# ---------------------------------------------------------------------------
@dataclass
class _ResultSnapshot:
    """Lightweight fingerprint of a single search result set."""

    image_hash: str
    top_domains: list[str]
    match_count: int
    timestamp: float


class ResultConsistencyTracker:
    """Tracks result stability for the same image across multiple runs."""

    def __init__(self) -> None:
        self._history: dict[str, deque[_ResultSnapshot]] = {}
        self._lock = Lock()

    def record(self, image_hash: str, top_domains: list[str], match_count: int) -> None:
        snapshot = _ResultSnapshot(
            image_hash=image_hash,
            top_domains=top_domains,
            match_count=match_count,
            timestamp=time.time(),
        )
        with self._lock:
            if image_hash not in self._history:
                self._history[image_hash] = deque(maxlen=_CONSISTENCY_HISTORY_LIMIT)
            self._history[image_hash].append(snapshot)

    def check_consistency(self, image_hash: str, current_domains: list[str]) -> str:
        """Compare current results to historical results.

        Returns:
            'stable'    — ≥60% overlap in top domains with previous runs
            'unstable'  — <40% overlap
            'first_run' — no previous data
            'partial'   — between thresholds
        """
        with self._lock:
            history = self._history.get(image_hash)
            if not history or len(history) < 1:
                return "first_run"

        previous = history[-1]
        if not previous.top_domains and not current_domains:
            return "stable"
        if not previous.top_domains or not current_domains:
            return "unstable"

        prev_set = set(previous.top_domains)
        curr_set = set(current_domains)
        union = prev_set | curr_set
        if not union:
            return "stable"

        overlap = len(prev_set & curr_set) / len(union)
        if overlap >= _CONSISTENCY_STABLE_THRESHOLD:
            return "stable"
        if overlap < _CONSISTENCY_UNSTABLE_THRESHOLD:
            return "unstable"
        return "partial"


# ---------------------------------------------------------------------------
# 6. Main Diagnostics Service (singleton)
# ---------------------------------------------------------------------------
class ReverseSearchDiagnostics:
    """Central diagnostics hub — thread-safe, singleton-friendly."""

    _instance: "ReverseSearchDiagnostics | None" = None
    _instance_lock = Lock()

    def __init__(self) -> None:
        self._audit_log: deque[dict[str, Any]] = deque(maxlen=_MAX_AUDIT_LOG_SIZE)
        self._request_timestamps: dict[str, deque[float]] = {}  # engine → timestamps
        self._lock = Lock()
        self._rate_limiters: dict[str, _TokenBucket] = {}
        self.consistency_tracker = ResultConsistencyTracker()

    @classmethod
    def get_instance(cls) -> "ReverseSearchDiagnostics":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # -- Rate Limiter Access ------------------------------------------------
    def get_rate_limiter(self, engine: str, rate_per_second: float = 1.0, burst: int = 3) -> _TokenBucket:
        with self._lock:
            if engine not in self._rate_limiters:
                self._rate_limiters[engine] = _TokenBucket(rate_per_second, burst)
            return self._rate_limiters[engine]

    # -- Input Fingerprinting -----------------------------------------------
    def log_input(self, fingerprint: SearchInputFingerprint) -> None:
        entry = {
            "type": "input",
            "timestamp": time.time(),
            **fingerprint.to_dict(),
        }
        LOGGER.info(
            "[SearchDiag] INPUT cid=%s hash=%s size=%d res=%dx%d mime=%s cloudinary=%s transforms=%s",
            fingerprint.correlation_id,
            fingerprint.file_hash_sha256[:16],
            fingerprint.file_size_bytes,
            fingerprint.width,
            fingerprint.height,
            fingerprint.mime_type,
            fingerprint.cloudinary_url or "<none>",
            fingerprint.transformations_applied,
        )
        with self._lock:
            self._audit_log.append(entry)

    # -- Request Tracking ---------------------------------------------------
    def log_request(self, log: SearchRequestLog) -> None:
        now = time.time()
        entry = {"type": "request", "timestamp": now, **log.to_dict()}
        LOGGER.info(
            "[SearchDiag] REQUEST cid=%s engine=%s url=%s attempt=%d",
            log.correlation_id,
            log.engine,
            log.image_url[:80],
            log.attempt_number,
        )
        with self._lock:
            self._audit_log.append(entry)
            # Track frequency per engine
            if log.engine not in self._request_timestamps:
                self._request_timestamps[log.engine] = deque(maxlen=200)
            self._request_timestamps[log.engine].append(now)
            # Burst detection
            engine_ts = self._request_timestamps[log.engine]
            recent = [t for t in engine_ts if now - t <= _BURST_WINDOW_SECONDS]
            if len(recent) > _BURST_THRESHOLD:
                LOGGER.warning(
                    "[SearchDiag] BURST DETECTED engine=%s requests_in_window=%d window=%.1fs",
                    log.engine,
                    len(recent),
                    _BURST_WINDOW_SECONDS,
                )

    # -- Response Tracking --------------------------------------------------
    def log_response(self, log: SearchResponseLog) -> None:
        entry = {"type": "response", "timestamp": time.time(), **log.to_dict()}
        level = logging.WARNING if log.quality_status in ("rate_limited", "failed") else logging.INFO
        LOGGER.log(
            level,
            "[SearchDiag] RESPONSE cid=%s engine=%s status=%s results=%d latency=%.0fms top3=%s error=%s",
            log.correlation_id,
            log.engine,
            log.quality_status,
            log.result_count,
            log.latency_ms,
            log.top_3_domains,
            log.raw_error or "<none>",
        )
        with self._lock:
            self._audit_log.append(entry)

    # -- Request Frequency --------------------------------------------------
    def requests_in_window(self, engine: str, window_seconds: float = _RATE_WINDOW_SECONDS) -> int:
        now = time.time()
        with self._lock:
            timestamps = self._request_timestamps.get(engine, deque())
            return sum(1 for t in timestamps if now - t <= window_seconds)

    # -- Audit Log Access ---------------------------------------------------
    def recent_audit_entries(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            entries = list(self._audit_log)
        return entries[-limit:]

    def audit_log_json(self, limit: int = 50) -> str:
        return json.dumps(self.recent_audit_entries(limit), indent=2, default=str)


# ---------------------------------------------------------------------------
# Module-level helpers for convenient access
# ---------------------------------------------------------------------------
def get_diagnostics() -> ReverseSearchDiagnostics:
    """Return the global diagnostics singleton."""
    return ReverseSearchDiagnostics.get_instance()


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def extract_top_domains(results: dict[str, Any], limit: int = 5) -> list[str]:
    """Pull the top N unique domains from a parsed result set."""
    domains: list[str] = []
    seen: set[str] = set()
    for section in ("top_matches", "similar_images", "visual_matches", "image_results"):
        for item in (results.get(section) or []):
            domain = item.get("domain") or ""
            if domain and domain not in seen:
                seen.add(domain)
                domains.append(domain)
            if len(domains) >= limit:
                return domains
    # Also check sources
    for source in (results.get("sources") or []):
        domain = source.get("domain") or ""
        if domain and domain not in seen:
            seen.add(domain)
            domains.append(domain)
        if len(domains) >= limit:
            return domains
    return domains
