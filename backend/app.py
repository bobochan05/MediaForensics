from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import requests
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from io import BytesIO
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from urllib.error import URLError
from urllib.parse import quote_plus, urlparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen
import urllib3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from flask import Flask, g, jsonify, make_response, redirect, render_template, request, url_for
from PIL import Image
from werkzeug.exceptions import HTTPException
from ai.layer2_matching.insights import build_layer2_insights

from backend.config import load_backend_config
from backend.auth_system import init_auth_system
from backend.auth_system.middleware import jwt_required
from backend.routes import parse_analyze_request, parse_chat_request
from backend.services import build_alerts, build_growth_indicator, build_layer1_payload, build_risk_insight

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["NO_PROXY"] = "*"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
urllib3.disable_warnings()
load_dotenv(PROJECT_ROOT / ".env")

APP_BUILD = "2026-04-09-layer2-debug"
RUNTIME_CONFIG = load_backend_config(PROJECT_ROOT)
INFERENCE_TIMEOUT_SECONDS = RUNTIME_CONFIG.inference_timeout_seconds
DASHBOARD_FRONTEND_URL = RUNTIME_CONFIG.dashboard_frontend_url
CORS_ALLOWED_ORIGINS = RUNTIME_CONFIG.cors_allowed_origins
REMOTE_PROXY_URL = str(os.getenv("REMOTE_PROXY_URL") or "").strip()
APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = RUNTIME_CONFIG.artifacts_dir

FUSION_PATH = ARTIFACTS_DIR / "models" / "fusion_model.pth"
EFFICIENTNET_PATH = ARTIFACTS_DIR / "models" / "efficientnet_finetuned.pth"
DINO_PATH = ARTIFACTS_DIR / "models" / "dino_finetuned.pth"
METRICS_PATH = ARTIFACTS_DIR / "outputs" / "metrics.json"
UI_UPLOADS_DIR = ARTIFACTS_DIR / "ui_uploads"
UI_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_INPUT_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
REMOTE_FETCH_TIMEOUT_SECONDS = 20
REMOTE_FETCH_MAX_BYTES = 50 * 1024 * 1024
REMOTE_CONTENT_TYPE_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
}
GENERIC_KEYWORDS = [
    "what is",
    "definition",
    "meaning of",
    "fictional character",
    "overview",
    "introduction",
    "history of",
]
GENERIC_DOMAINS = ["wikipedia", "britannica", "dictionary"]
EXACT_VISUAL_THRESHOLD = 0.85
RELATED_VISUAL_THRESHOLD = 0.5
VISUAL_LOOKALIKE_THRESHOLD = 0.55
EMBEDDING_MATCH_THRESHOLD = 0.6
INTERNAL_EMBEDDING_MIN_SCORE = 0.55
INTERNAL_EMBEDDING_MAX_VISUAL_SIMILARITY = 0.82
EXACT_PHASH_THRESHOLD = 10
DISCOVERY_SECTION_LIMIT = 10
INTERNAL_EMBEDDING_SEARCH_LIMIT = 100

app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
app.config["MAX_CONTENT_LENGTH"] = RUNTIME_CONFIG.max_upload_size_bytes
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["SECRET_KEY"] = (os.getenv("FLASK_SECRET_KEY") or os.urandom(32).hex())

_LAYER2_PIPELINE = None
LOGGER = logging.getLogger(__name__)
AUTH_SERVICE = init_auth_system(app, PROJECT_ROOT)
_LAYER2_RESULT_STORE: dict[str, dict[str, object]] = {}
_LAYER2_RESULT_STORE_META: dict[str, float] = {}
_LAYER2_RESULT_STORE_LOCK = Lock()
_LAYER2_STORE_TTL_SECONDS = 3600
REQUEST_TIMEOUT_SECONDS = 15.0
REVERSE_SEARCH_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
REVERSE_SEARCH_ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _current_request_id() -> str:
    request_id = str(getattr(g, "request_id", "") or "").strip()
    return request_id or _new_uuid()


def _layer2_match_similarity(item: dict[str, object]) -> float:
    metadata = dict(item.get("metadata") or {})
    score_breakdown = dict(item.get("score_breakdown") or {})
    similarity = _safe_float(
        score_breakdown.get("final_score"),
        _safe_float(
            metadata.get("embedding_similarity"),
            _safe_float(
                item.get("visual_similarity"),
                _safe_float(
                    item.get("fused_similarity"),
                    _safe_float(metadata.get("final_score")),
                ),
            ),
        ),
    )
    return max(0.0, min(1.0, similarity))


def _layer2_match_preview_url(item: dict[str, object]) -> str:
    metadata = dict(item.get("metadata") or {})
    candidate_urls = [
        item.get("preview_url"),
        item.get("media_url"),
        metadata.get("media_url"),
        metadata.get("resolved_media_url"),
        metadata.get("provider_media_url"),
        metadata.get("thumbnail"),
        metadata.get("provider_image_url"),
        item.get("image_url"),
    ]
    candidate_urls.extend(metadata.get("resolved_image_urls") or [])
    source_url = str(item.get("url") or item.get("source_url") or "").strip()
    if source_url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        candidate_urls.append(source_url)

    for candidate in candidate_urls:
        cleaned = str(candidate or "").strip()
        if cleaned.startswith(("http://", "https://")):
            return cleaned
    return ""


def _layer2_display_matches(
    exact_matches: list[dict[str, object]],
    visual_matches_top10: list[dict[str, object]],
    embedding_matches_top10: list[dict[str, object]],
) -> list[dict[str, object]]:
    display_items = [*exact_matches, *visual_matches_top10, *embedding_matches_top10]
    matches: list[dict[str, object]] = []
    seen_keys: set[str] = set()

    for item in display_items:
        source_url = str(item.get("url") or item.get("page_url") or item.get("source_url") or "").strip()
        preview_url = _layer2_match_preview_url(item)
        identity = source_url or preview_url or str(item.get("id") or item.get("entry_id") or item.get("candidate_key") or "").strip()
        if identity and identity in seen_keys:
            continue
        if identity:
            seen_keys.add(identity)

        title = str(item.get("title") or "").strip()
        if not title:
            parsed = urlparse(source_url)
            title = Path(parsed.path).name or f"Source match {len(matches) + 1}"

        matches.append(
            {
                "id": str(item.get("id") or item.get("entry_id") or item.get("candidate_key") or f"match-{len(matches) + 1}"),
                "preview_url": preview_url,
                "source_url": source_url or preview_url,
                "similarity": _layer2_match_similarity(item),
                "first_seen": str(item.get("timestamp") or item.get("first_seen") or "").strip(),
                "platform": str(item.get("platform") or "web"),
                "title": title,
            }
        )
        if len(matches) >= DISCOVERY_SECTION_LIMIT * 3:
            break

    return matches


def _match_domain(item: dict[str, object]) -> str:
    source_url = str(
        item.get("page_url")
        or item.get("url")
        or item.get("source_url")
        or item.get("media_url")
        or ""
    ).strip()
    if not source_url:
        return "unknown"
    parsed = urlparse(source_url)
    return parsed.netloc.lower().removeprefix("www.") or "unknown"


def _match_type_label(item: dict[str, object], default_type: str) -> str:
    raw_type = str(item.get("type") or "").strip().lower()
    if raw_type == "exact_match":
        return "exact"
    if raw_type == "visually_similar":
        return "visual"
    if raw_type in {"embedding_similar", "related_content"}:
        return "semantic"
    return default_type


def _match_explanation(item: dict[str, object], match_type: str) -> str:
    note = str(item.get("note") or item.get("match_reason") or "").strip()
    if note:
        return note
    if match_type == "exact":
        return "High pixel similarity and corroborated reverse-search evidence."
    if match_type == "visual":
        return "Similar composition, framing, or subject appearance."
    return "Contextually related result surfaced by reverse-image search."


def _annotate_matches(items: list[dict[str, object]], *, default_type: str) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for item in items:
        enriched = deepcopy(item)
        match_type = _match_type_label(enriched, default_type)
        enriched["match_type"] = match_type
        enriched["similarity_score"] = round(_layer2_match_similarity(enriched), 4)
        enriched["explanation"] = _match_explanation(enriched, match_type)
        enriched["domain"] = _match_domain(enriched)
        annotated.append(enriched)
    return annotated


def _top_domains_from_matches(items: list[dict[str, object]], limit: int = 3) -> list[str]:
    counts: dict[str, int] = {}
    for item in items:
        domain = str(item.get("domain") or _match_domain(item)).strip().lower()
        if not domain or domain == "unknown":
            continue
        counts[domain] = counts.get(domain, 0) + 1
    return [domain for domain, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]


def _first_seen_estimate(items: list[dict[str, object]]) -> str:
    timestamps = [
        str(item.get("timestamp") or item.get("first_seen") or "").strip()
        for item in items
        if str(item.get("timestamp") or item.get("first_seen") or "").strip()
    ]
    if not timestamps:
        return "Unavailable from current providers"
    return min(timestamps)


def _best_origin_match(
    exact_matches: list[dict[str, object]],
    visual_matches: list[dict[str, object]],
    related_sources: list[dict[str, object]],
) -> dict[str, object] | None:
    for bucket in (exact_matches, visual_matches, related_sources):
        if bucket:
            return sorted(bucket, key=lambda item: item.get("similarity_score") or 0.0, reverse=True)[0]
    return None


def _domain_clusters(items: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in items:
        domain = str(item.get("domain") or _match_domain(item)).strip().lower() or "unknown"
        grouped.setdefault(domain, []).append(item)
    return dict(sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])))


def _origin_summary_payload(
    exact_matches: list[dict[str, object]],
    visual_matches: list[dict[str, object]],
    related_sources: list[dict[str, object]],
) -> dict[str, object]:
    all_matches = [*exact_matches, *visual_matches, *related_sources]
    top_domains = _top_domains_from_matches(all_matches)
    strongest = _best_origin_match(exact_matches, visual_matches, related_sources)
    source_count = len(all_matches)
    strongest_domain = str((strongest or {}).get("domain") or "an unknown source").strip()
    first_seen = _first_seen_estimate(all_matches)
    if top_domains:
        domain_story = ", ".join(top_domains[:3])
        summary = (
            f"This image most likely originated from {strongest_domain} and has appeared across {source_count} sources. "
            f"Strong matches were found on {domain_story}. The content appears to be reused or modified across multiple platforms."
        )
    else:
        summary = (
            f"This image most likely originated from {strongest_domain} and has appeared across {source_count} sources. "
            "Provider data is limited, so this origin estimate should be reviewed manually."
        )
    return {
        "origin_summary": summary,
        "top_domains": top_domains,
        "first_seen_estimate": first_seen,
    }


def _consistency_warning_payload(layer1_result: str, layer1_confidence: float, exact_matches: list[dict[str, object]]) -> dict[str, object]:
    has_strong_exact = any(float(item.get("similarity_score") or 0.0) >= 0.85 for item in exact_matches)
    enabled = str(layer1_result or "").upper() == "FAKE" and float(layer1_confidence or 0.0) >= 80.0 and has_strong_exact
    message = (
        "AI model flags this as synthetic, but strong real-world matches exist. Review manually."
        if enabled
        else ""
    )
    return {
        "consistency_warning": enabled,
        "message": message,
    }


def build_layer2_response(data: dict[str, object] | None) -> dict[str, object]:
    payload = data if isinstance(data, dict) else {}
    exact_raw = payload.get("exact_matches")
    visual_raw = payload.get("visual_matches_top10")
    embedding_raw = payload.get("related_web_sources")
    if not isinstance(embedding_raw, list):
        embedding_raw = payload.get("embedding_matches_top10")
    exact = _annotate_matches(list(exact_raw) if isinstance(exact_raw, list) else [], default_type="exact")
    visual = _annotate_matches(list(visual_raw) if isinstance(visual_raw, list) else [], default_type="visual")
    embedding = _annotate_matches(list(embedding_raw) if isinstance(embedding_raw, list) else [], default_type="semantic")
    matches = _layer2_display_matches(exact, visual, embedding)
    origin_payload = _origin_summary_payload(exact, visual, embedding)
    domain_clusters = _domain_clusters([*exact, *visual, *embedding])
    response = {
        "exact_matches": exact,
        "visual_matches_top10": visual,
        "embedding_matches_top10": embedding,
        "related_web_sources": embedding,
        "visual_matches": visual,
        "embedding_matches": embedding,
        "counts": {
            "exact": len(exact),
            "visual": len(visual),
            "embedding": len(embedding),
        },
        "matches": matches,
        "count": len(matches),
        "origin_summary": origin_payload["origin_summary"],
        "top_domains": origin_payload["top_domains"],
        "first_seen_estimate": origin_payload["first_seen_estimate"],
        "domain_clusters": domain_clusters,
    }
    for key in (
        "spread_analysis",
        "provider_status",
        "image_url",
        "message",
        "status",
        "execution",
        "fallback_used",
        "confidence_score",
        "context_analysis",
        "clusters",
        "errors",
        "sources",
        "manual_search_links",
        "manual_search_note",
        "reverse_search",
        "consistency_warning",
    ):
        if key in payload:
            response[key] = deepcopy(payload[key])
    return response


def _cleanup_layer2_store() -> None:
    now = time.time()
    expired_keys: list[str] = []
    with _LAYER2_RESULT_STORE_LOCK:
        for key, created_at in list(_LAYER2_RESULT_STORE_META.items()):
            if now - float(created_at) > _LAYER2_STORE_TTL_SECONDS:
                expired_keys.append(key)
        for key in expired_keys:
            _LAYER2_RESULT_STORE.pop(key, None)
            _LAYER2_RESULT_STORE_META.pop(key, None)


def _build_layer2_channels(
    exact_matches: list[dict[str, object]] | None = None,
    visual_matches_top10: list[dict[str, object]] | None = None,
    embedding_matches_top10: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return build_layer2_response(
        {
            "exact_matches": list(exact_matches or []),
            "visual_matches_top10": list(visual_matches_top10 or []),
            "embedding_matches_top10": list(embedding_matches_top10 or []),
        }
    )


def _store_layer2_channels(upload_id: str, layer2_payload: dict[str, object]) -> None:
    if not upload_id:
        return
    _cleanup_layer2_store()
    normalized_layer2 = build_layer2_response(layer2_payload)
    with _LAYER2_RESULT_STORE_LOCK:
        _LAYER2_RESULT_STORE[upload_id] = deepcopy(normalized_layer2)
        _LAYER2_RESULT_STORE_META[upload_id] = time.time()


def _get_layer2_channels(upload_id: str) -> dict[str, object] | None:
    if not upload_id:
        return None
    _cleanup_layer2_store()
    with _LAYER2_RESULT_STORE_LOCK:
        payload = _LAYER2_RESULT_STORE.get(upload_id)
        if payload is None:
            return None
        return build_layer2_response(deepcopy(payload))


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _model_paths() -> list[dict[str, object]]:
    model_paths = [
        {"label": "Fusion MLP", "path": FUSION_PATH},
        {"label": "EfficientNet", "path": EFFICIENTNET_PATH},
        {"label": "DINO", "path": DINO_PATH},
    ]
    return [
        {
            "label": entry["label"],
            "path": _display_path(entry["path"]),
            "exists": Path(entry["path"]).exists(),
        }
        for entry in model_paths
    ]


def _missing_models() -> list[str]:
    return [entry["path"] for entry in _model_paths() if not entry["exists"]]


def _template_context(*, service_entry: bool = False) -> dict[str, object]:
    model_paths = _model_paths()
    auth_context = AUTH_SERVICE.build_template_context(request)
    return {
        "app_build": APP_BUILD,
        "model_paths": model_paths,
        "missing_models": [entry["path"] for entry in model_paths if not entry["exists"]],
        "metrics": _load_metrics(),
        "service_entry": service_entry,
        "auth_state": auth_context["auth_state"],
        "has_service_access": auth_context["has_service_access"],
        "user_email": auth_context["user_email"],
        "user_username": auth_context["user_username"],
        "guest_limit": auth_context["guest_limit"],
        "guest_used": auth_context["guest_used"],
        "guest_remaining": auth_context["guest_remaining"],
        "google_oauth_client_id": AUTH_SERVICE.settings.google_client_id or "",
    }


def _load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None

    try:
        with METRICS_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _inference_python() -> str:
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _run_inference_subprocess(input_path: Path) -> tuple[str, float]:
    command = [
        _inference_python(),
        "-m",
        "ai.layer1_detection.inference",
        "--input_path",
        str(input_path),
        "--classifier_path",
        str(FUSION_PATH),
        "--efficientnet_path",
        str(EFFICIENTNET_PATH),
        "--dino_path",
        str(DINO_PATH),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Inference timed out after {INFERENCE_TIMEOUT_SECONDS} seconds.") from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or "Unknown inference error."
        raise RuntimeError(detail)

    stdout = completed.stdout or ""
    label_match = re.search(r"^Prediction:\s*(real|fake)\s*$", stdout, flags=re.IGNORECASE | re.MULTILINE)
    confidence_match = re.search(r"^Confidence:\s*([0-9]*\.?[0-9]+)\s*$", stdout, flags=re.MULTILINE)
    if not label_match or not confidence_match:
        raise RuntimeError(f"Unexpected inference output:\n{stdout.strip()}")

    return label_match.group(1).lower(), float(confidence_match.group(1))


def _upload_meta_path(upload_id: str) -> Path:
    return UI_UPLOADS_DIR / f"{upload_id}.json"


def _save_uploaded_file(uploaded_file) -> tuple[str, Path]:
    upload_id = _new_uuid()
    suffix = Path(uploaded_file.filename or "").suffix.lower() or ".jpg"
    stored_path = UI_UPLOADS_DIR / f"{upload_id}{suffix}"
    uploaded_file.save(stored_path)
    return upload_id, stored_path


def _remote_filename_from_url(source_url: str) -> str:
    parsed = urlparse(source_url)
    name = Path(parsed.path).name.strip()
    return name or "remote_media"


def _download_remote_media(source_url: str) -> tuple[str, Path, str]:
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Source URL must start with http:// or https://")

    req = UrlRequest(source_url, headers={"User-Agent": "Tracelyt/1.0"})
    try:
        with urlopen(req, timeout=REMOTE_FETCH_TIMEOUT_SECONDS) as response:
            content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            if content_type and not (content_type.startswith("image/") or content_type.startswith("video/")):
                raise ValueError("Source URL must point to an image or video file.")

            filename = _remote_filename_from_url(source_url)
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_INPUT_SUFFIXES:
                suffix = REMOTE_CONTENT_TYPE_SUFFIX.get(content_type, "")
            if suffix not in ALLOWED_INPUT_SUFFIXES:
                guessed = mimetypes.guess_extension(content_type or "") or ""
                suffix = guessed.lower() if guessed.lower() in ALLOWED_INPUT_SUFFIXES else ""
            if suffix not in ALLOWED_INPUT_SUFFIXES:
                raise ValueError("Unsupported source URL format. Use JPG, PNG, MP4, AVI, MOV, MKV, or WEBM.")

            upload_id = _new_uuid()
            stored_path = UI_UPLOADS_DIR / f"{upload_id}{suffix}"
            bytes_written = 0
            with stored_path.open("wb") as output:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > REMOTE_FETCH_MAX_BYTES:
                        stored_path.unlink(missing_ok=True)
                        raise ValueError("Source URL file is too large. Keep it under 50 MB.")
                    output.write(chunk)

            if bytes_written == 0:
                stored_path.unlink(missing_ok=True)
                raise ValueError("Could not download media from the source URL.")

            original_name = filename if Path(filename).suffix else f"{filename}{suffix}"
            return upload_id, stored_path, original_name
    except URLError as exc:
        raise ValueError("Could not fetch media from the provided URL.") from exc


def _store_upload_metadata(
    upload_id: str,
    stored_path: Path,
    original_filename: str,
    label: str,
    confidence: float,
    source_url: str | None = None,
) -> None:
    payload = {
        "upload_id": upload_id,
        "stored_path": str(stored_path),
        "original_filename": original_filename,
        "label": label,
        "confidence": confidence,
        "source_url": source_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _upload_meta_path(upload_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_upload_metadata(upload_id: str) -> dict[str, object]:
    metadata_path = _upload_meta_path(upload_id)
    if not metadata_path.exists():
        raise FileNotFoundError("That analysis session was not found. Please upload the file again.")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _write_upload_metadata(upload_id: str, payload: dict[str, object]) -> None:
    _upload_meta_path(upload_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ensure_public_source_url(upload_id: str, upload_metadata: dict[str, object]) -> str | None:
    source_url = str(upload_metadata.get("source_url") or "").strip()
    if source_url.startswith(("http://", "https://")):
        return source_url

    stored_path = Path(str(upload_metadata.get("stored_path") or ""))
    if not stored_path.exists():
        return None

    try:
        upload_target = stored_path
        original_filename = str(upload_metadata.get("original_filename") or stored_path.name)
        if stored_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            from ai.layer1_detection.frame_extractor import extract_sampled_frames

            preview_dir = UI_UPLOADS_DIR / "preview_frames"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / f"{upload_id}.jpg"
            if not preview_path.exists():
                frames = extract_sampled_frames(
                    video_path=stored_path,
                    image_size=512,
                    sample_fps=0.5,
                    frames_per_video=1,
                )
                if not frames:
                    return None
                frames[0].save(preview_path, format="JPEG", quality=92)
            upload_target = preview_path
            original_filename = f"{Path(original_filename).stem}.jpg"

        from ai.layer2_matching.tracking.reverse_search_service import upload_path_to_cloudinary

        public_url = upload_path_to_cloudinary(
            upload_target,
            filename_override=original_filename,
        )
    except Exception:
        LOGGER.exception("Failed to create a public source URL for %s", stored_path)
        return None

    updated_metadata = dict(upload_metadata)
    updated_metadata["source_url"] = public_url
    updated_metadata["source_url_generated"] = True
    _write_upload_metadata(upload_id, updated_metadata)
    return public_url


def _certainty_band(confidence: float) -> str:
    if confidence >= 0.9:
        return "high"
    if confidence >= 0.75:
        return "moderate"
    return "low"


def _confidence_explainer(certainty: str, label: str) -> str:
    leaning = "fake" if label == "fake" else "real"
    if certainty == "high":
        return (
            f"The score is comfortably away from the decision boundary, so the model is showing a stronger {leaning.upper()} "
            "preference than it would in a borderline case."
        )
    if certainty == "moderate":
        return (
            "The model has a noticeable preference, but this still sits in the zone where a human review is useful, "
            "especially if the content is high-stakes or politically sensitive."
        )
    return (
        "This is a lower-certainty decision. The model still produced a verdict, but the final score is relatively close "
        "to the cutoff, so it should be treated as a signal that deserves extra caution."
    )


def _build_reasoning(label: str, confidence: float, original_filename: str) -> dict[str, object]:
    label = label.lower()
    fake_probability = confidence if label == "fake" else 1.0 - confidence
    confidence_pct = round(confidence * 100, 2)
    fake_probability_pct = round(fake_probability * 100, 2)
    media_suffix = Path(original_filename).suffix.lower()
    media_type = "video" if media_suffix in VIDEO_SUFFIXES else "image"
    certainty = _certainty_band(confidence)

    if label == "fake":
        headline = "The detector marked this content as FAKE because the final fake score crossed the model's decision boundary."
        summary = (
            f"This {media_type} was classified as FAKE after the combined model estimated a fake probability of "
            f"{fake_probability_pct}%. That is above the 50% threshold the system uses to separate likely manipulated "
            "content from likely authentic content."
        )
        decision_logic = (
            "In plain language, the system saw more patterns that resemble synthetic or manipulated media than patterns "
            "that resemble authentic capture. That conclusion comes from the final fused score, not from a single isolated clue."
        )
    else:
        headline = "The detector marked this content as REAL because the final fake score stayed below the model's decision boundary."
        summary = (
            f"This {media_type} was classified as REAL because the estimated fake probability remained at "
            f"{fake_probability_pct}%, which is below the 50% threshold used by the final fusion model."
        )
        decision_logic = (
            "In plain language, the system did not find enough evidence of manipulation to push the final score into the "
            "fake range. That does not prove the content is genuine with certainty, but it means the model leaned toward authenticity."
        )

    processing_story = (
        f"The system first normalized the {media_type} into the format expected by Layer 1. "
        + (
            "For video, it samples frames at 0.5 FPS so the detector can inspect representative moments without processing every frame. "
            if media_type == "video"
            else "For images, it runs the media through the same preprocessing stack used during training so the learned features stay consistent. "
        )
        + "Those processed views are then passed through multiple branches that look at the content in different ways before a final fusion model makes the verdict."
    )
    model_story = (
        "The verdict is based on a combined signal from CLIP, EfficientNet, DINO, and FFT-based features. "
        "That means the decision reflects semantic cues, visual texture cues, transformer-style representations, and frequency-domain artifacts together, "
        "instead of relying on just one type of evidence."
    )
    confidence_story = _confidence_explainer(certainty, label)
    caution = (
        "This explanation describes how the model reached its score. It should be read as model reasoning, not courtroom proof. "
        "If the content could cause harm, misinformation, or reputational damage, a manual review is still the safest next step."
    )
    factors = [
        f"The file was handled as a {media_type}, then normalized before inference so it matched the training and inference pipeline.",
        "The final score came from a fusion model that combines multiple feature streams rather than trusting a single network.",
        "Confidence here means distance from the 50% cutoff, so it indicates decisiveness of the model, not an absolute guarantee.",
    ]
    steps = [
        f"Step 1: preprocess the {media_type} and prepare the frames or image tensors for the detector.",
        "Step 2: extract complementary signals using CLIP, EfficientNet, DINO, and FFT-informed features.",
        "Step 3: combine those signals in the fusion classifier to estimate the fake probability.",
        "Step 4: compare the final probability against the 50% decision threshold to produce the final label.",
    ]

    return {
        "headline": headline,
        "summary": summary,
        "decision_logic": decision_logic,
        "processing_story": processing_story,
        "model_story": model_story,
        "confidence_story": confidence_story,
        "caution": caution,
        "confidence_percent": confidence_pct,
        "fake_probability_percent": fake_probability_pct,
        "decision_threshold_percent": 50.0,
        "certainty_band": certainty,
        "factors": factors,
        "steps": steps,
    }


def _manual_search_links(
    original_filename: str,
    query_hint: str | None = None,
    source_url: str | None = None,
) -> tuple[list[dict[str, str]], str]:
    stem = Path(original_filename).stem.replace("_", " ").replace("-", " ").strip()
    query = query_hint.strip() if query_hint and query_hint.strip() else stem
    encoded_query = quote_plus(query)
    cleaned_source_url = str(source_url or "").strip()
    public_source_url = cleaned_source_url if cleaned_source_url.startswith(("http://", "https://")) else None

    if public_source_url:
        encoded_source_url = quote_plus(public_source_url)
        return (
            [
                {"label": "Open Google Lens", "url": f"https://lens.google.com/uploadbyurl?url={encoded_source_url}"},
                {"label": "Open Bing Visual Search", "url": f"https://www.bing.com/images/searchbyimage?cbir=sbi&imgurl={encoded_source_url}"},
                {"label": "Open Yandex Images", "url": f"https://yandex.com/images/search?rpt=imageview&url={encoded_source_url}"},
                {"label": "Search The Web", "url": f"https://www.google.com/search?q={encoded_query}"},
            ],
            "These buttons use an available public image URL, so the image engines can run a real reverse-image search.",
        )

    return (
        [
            {"label": "Open Google Lens", "url": "https://lens.google.com/"},
            {"label": "Open Bing Visual Search", "url": "https://www.bing.com/visualsearch"},
            {"label": "Open Yandex Images", "url": "https://yandex.com/images/"},
            {"label": "Search The Web", "url": f"https://www.google.com/search?q={encoded_query}"},
        ],
        "No public image URL was available, so the image buttons open each engine's upload page instead of searching by the local filename.",
    )


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_generic_page(item: dict[str, object]) -> bool:
    title = str(item.get("title") or "").lower()
    caption = str(item.get("caption") or "").lower()
    url = str(item.get("url") or "").lower()
    if any(keyword in title or keyword in caption for keyword in GENERIC_KEYWORDS):
        return True
    return any(domain in url for domain in GENERIC_DOMAINS)


def _result_confidence(item_type: str, similarity: float, num_sources: int, num_frames: int, phash_diff: int | None) -> str:
    if item_type == "exact_match":
        if similarity >= 0.92 and num_sources >= 2 and num_frames >= 2 and (phash_diff is not None and phash_diff <= 8):
            return "HIGH"
        return "MEDIUM"
    if item_type == "embedding_similar":
        if similarity >= 0.7:
            return "MEDIUM"
        return "LOW"
    if item_type == "visually_similar":
        if similarity >= 0.9 and (num_sources >= 2 or num_frames >= 2):
            return "MEDIUM"
        return "LOW"
    if similarity >= 0.7 and (num_sources >= 2 or num_frames >= 2):
        return "MEDIUM"
    return "LOW"


def _result_note(item_type: str, confidence: str) -> str:
    if item_type == "exact_match":
        return "Near-duplicate image found" if confidence == "HIGH" else "Strong visual match found"
    if item_type == "embedding_similar":
        return "External semantic or context-based lead from reverse-image discovery. Useful for investigation when there is no strong direct visual lookalike."
    if item_type == "visually_similar":
        return "A downloaded image or frame from this result is visually similar to your upload. This is useful for lookalike discovery, not proof of the original source."
    return "Embedding-based lead from page context or broader semantic similarity, not a direct visual lookalike."


def _normalized_url_key(url: str | None) -> str:
    cleaned = str(url or "").strip()
    if not cleaned:
        return ""
    parsed = urlparse(cleaned)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}"
    return cleaned.rstrip("/").lower()


def _reverse_search_asset_key(url: str | None) -> str:
    cleaned = str(url or "").strip()
    if not cleaned:
        return ""
    parsed = urlparse(cleaned)
    normalized_path = parsed.path.lower()
    if "/deepfake-detector/reverse-search/" not in normalized_path:
        return ""
    stem = Path(normalized_path).stem
    return re.sub(r"_[a-z0-9]{5,}$", "", stem)


def _domain_label(url: str | None) -> str:
    cleaned = str(url or "").strip()
    if not cleaned:
        return "web"
    parsed = urlparse(cleaned)
    return parsed.netloc.lower().removeprefix("www.") or "web"


def _reverse_rank_score(bucket: str, rank: int) -> float:
    bucket_key = str(bucket or "").strip().lower()
    base_scores = {
        "visual_matches": 0.8,
        "top_matches": 0.72,
        "image_results": 0.66,
        "related_content": 0.58,
        "inline_images": 0.54,
        "similar_images": 0.54,
    }
    decay = 0.028 if bucket_key == "visual_matches" else 0.024
    base = base_scores.get(bucket_key, 0.6)
    return max(0.28, min(0.94, base - (max(rank - 1, 0) * decay)))


def _serialize_public_web_result(
    item: dict[str, object],
    *,
    item_type: str,
    note: str,
) -> dict[str, object] | None:
    page_url = str(item.get("page_url") or item.get("link") or item.get("url") or "").strip()
    image_url = str(item.get("image_url") or item.get("image") or item.get("thumbnail") or "").strip()
    source_url = page_url if page_url.startswith(("http://", "https://")) else ""
    preview_url = image_url if image_url.startswith(("http://", "https://")) else ""
    if not source_url and not preview_url:
        return None

    bucket = str(item.get("bucket") or "").strip().lower()
    rank = int(_safe_float(item.get("rank"), 9999))
    trusted_domain = bool(item.get("trusted_domain"))
    domain = str(item.get("domain") or _domain_label(source_url or preview_url)).strip().lower() or "web"
    title = str(item.get("title") or "").strip() or domain
    caption = str(item.get("snippet") or item.get("source") or "").strip() or None
    final_score = _reverse_rank_score(bucket, rank)
    confidence = "MEDIUM" if bucket == "visual_matches" and rank <= 3 else "LOW"
    similarity = final_score if item_type == "visually_similar" else None
    credibility = 0.8 if trusted_domain else 0.5

    return {
        "id": source_url or preview_url or f"{bucket}-{rank}",
        "url": source_url or preview_url,
        "page_url": source_url or None,
        "preview_url": preview_url or None,
        "media_url": preview_url or None,
        "local_path": None,
        "visual_similarity": similarity,
        "audio_similarity": None,
        "fused_similarity": final_score,
        "combined_score": final_score,
        "platform": domain,
        "timestamp": None,
        "source_type": "external",
        "title": title,
        "caption": caption,
        "context": "web",
        "context_scores": {"web": 1.0},
        "credibility_score": credibility,
        "is_mock": False,
        "label": None,
        "metadata": {
            "provider": "serpapi_reverse_search",
            "reverse_bucket": bucket,
            "rank": rank,
            "trusted_domain": trusted_domain,
            "page_url": source_url or None,
            "image_url": preview_url or None,
            "thumbnail": preview_url or None,
            "final_score": final_score,
            "match_reason": note,
            "score_components": {
                "similarity": similarity if similarity is not None else final_score,
                "source_trust": credibility,
                "evidence": 0.4 if bucket == "visual_matches" else 0.25,
                "final_score": final_score,
            },
            "resolver_status": "public_web_result",
        },
        "type": item_type,
        "confidence": confidence,
        "note": note,
        "confidence_level": confidence,
        "match_reason": note,
        "resolver_status": "public_web_result",
        "score_breakdown": {
            "similarity": similarity if similarity is not None else final_score,
            "trust": credibility,
            "evidence": 0.4 if bucket == "visual_matches" else 0.25,
            "final_score": final_score,
        },
    }


def _merge_section_items(*groups: list[dict[str, object]], limit: int = DISCOVERY_SECTION_LIMIT) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    seen_keys: set[str] = set()

    for group in groups:
        for item in group:
            source_key = _normalized_url_key(
                str(item.get("page_url") or item.get("url") or item.get("source_url") or "")
            )
            preview_key = _normalized_url_key(
                str(item.get("preview_url") or item.get("media_url") or ((item.get("metadata") or {}).get("image_url")) or "")
            )
            identity = source_key or preview_key or str(item.get("id") or "").strip()
            if not identity or identity in seen_keys:
                continue
            seen_keys.add(identity)
            merged.append(item)
            if len(merged) >= limit:
                return merged

    return merged


def _reverse_visual_matches(
    reverse_search: dict[str, object],
    *,
    excluded_urls: set[str] | None = None,
    limit: int = DISCOVERY_SECTION_LIMIT,
) -> list[dict[str, object]]:
    excluded = {_normalized_url_key(url) for url in (excluded_urls or set()) if _normalized_url_key(url)}
    visual_matches: list[dict[str, object]] = []

    for raw_item in list(reverse_search.get("visual_matches") or []):
        serialized = _serialize_public_web_result(
            raw_item,
            item_type="visually_similar",
            note="Public lookalike result returned by reverse-image search. This is useful for independent verification and source hunting.",
        )
        if serialized is None:
            continue
        source_key = _normalized_url_key(str(serialized.get("page_url") or serialized.get("url") or ""))
        if source_key and source_key in excluded:
            continue
        visual_matches.append(serialized)
        if len(visual_matches) >= limit:
            break

    return visual_matches


def _related_web_source_matches(
    reverse_search: dict[str, object],
    *,
    excluded_urls: set[str] | None = None,
    limit: int = DISCOVERY_SECTION_LIMIT,
) -> list[dict[str, object]]:
    excluded = {_normalized_url_key(url) for url in (excluded_urls or set()) if _normalized_url_key(url)}
    related_matches: list[dict[str, object]] = []
    groups = [
        list(reverse_search.get("top_matches") or []),
        list(reverse_search.get("image_results") or []),
        list(reverse_search.get("similar_images") or []),
    ]

    for group in groups:
        for raw_item in group:
            serialized = _serialize_public_web_result(
                raw_item,
                item_type="embedding_similar",
                note="Related public web page returned by reverse-image search. Use this as a public lead for investigation, not as proof of originality.",
            )
            if serialized is None:
                continue
            source_key = _normalized_url_key(str(serialized.get("page_url") or serialized.get("url") or ""))
            if source_key and source_key in excluded:
                continue
            excluded.add(source_key)
            related_matches.append(serialized)
            if len(related_matches) >= limit:
                return related_matches

    return related_matches


def _item_identity(item) -> str:
    payload = item if isinstance(item, dict) else item.model_dump(mode="json")
    metadata = dict(payload.get("metadata") or {})
    for candidate in (
        payload.get("entry_id"),
        payload.get("url"),
        metadata.get("media_url"),
        metadata.get("downloaded_path"),
        payload.get("local_path"),
        payload.get("id"),
    ):
        value = str(candidate or "").strip()
        if value:
            return value
    return ""


def _item_payload(item) -> dict[str, object]:
    return item if isinstance(item, dict) else item.model_dump(mode="json")


def _is_media_backed_visual_candidate(item) -> bool:
    payload = _item_payload(item)
    metadata = dict(payload.get("metadata") or {})
    raw_similarity = _safe_float(metadata.get("raw_visual_similarity"), _safe_float(payload.get("visual_similarity")))
    if raw_similarity < VISUAL_LOOKALIKE_THRESHOLD:
        return False

    source_type = str(payload.get("source_type") or "").lower()
    provider = str(metadata.get("provider") or "").lower()
    downloaded_path = str(metadata.get("downloaded_path") or "").strip()
    media_url = str(metadata.get("media_url") or "").strip()
    resolved_media_url = str(metadata.get("resolved_media_url") or "").strip()
    resolved_media_type = str(metadata.get("resolved_media_type") or "").lower().strip()
    source_url = str(metadata.get("source_url") or "").strip()
    resolved_image_urls = [
        str(url).strip()
        for url in (metadata.get("resolved_image_urls") or [])
        if str(url).strip()
    ]

    query_url_key = _normalized_url_key(source_url)
    media_url_key = _normalized_url_key(resolved_media_url or media_url)
    query_asset_key = _reverse_search_asset_key(source_url)
    media_asset_key = _reverse_search_asset_key(resolved_media_url or media_url)
    candidate_page_has_distinct_media = any(
        _normalized_url_key(url) and _normalized_url_key(url) != query_url_key
        for url in resolved_image_urls
    )
    is_query_self_match = bool(
        (query_url_key and media_url_key and media_url_key == query_url_key)
        or (query_asset_key and media_asset_key and query_asset_key == media_asset_key)
    )

    has_external_media = (
        source_type == "external"
        and provider == "reverse_search"
        and downloaded_path
        and (not resolved_media_type or resolved_media_type in {"image", "video"})
        and not _is_generic_page(payload)
        and not is_query_self_match
        and (candidate_page_has_distinct_media or bool(media_url_key))
    )
    has_internal_media = source_type != "external" and bool(str(payload.get("local_path") or "").strip())
    return has_external_media or has_internal_media


def _resolver_status(item) -> str:
    payload = _item_payload(item)
    metadata = dict(payload.get("metadata") or {})
    explicit_status = str(payload.get("resolver_status") or metadata.get("resolver_status") or "").strip().lower()
    if explicit_status:
        return explicit_status
    return "ok" if _is_media_backed_visual_candidate(item) else "invalid"


def _embedding_similarity_value(item) -> float:
    payload = _item_payload(item)
    metadata = dict(payload.get("metadata") or {})
    return _safe_float(
        payload.get("embedding_similarity"),
        _safe_float(
            metadata.get("embedding_similarity"),
            _safe_float(
                metadata.get("embedding_rank_score"),
                _safe_float(payload.get("fused_similarity")),
            ),
        ),
    )


def _serialize_discovery_item(item, item_type: str) -> dict[str, object]:
    payload = _item_payload(item)
    metadata = dict(payload.get("metadata") or {})
    raw_similarity = _safe_float(metadata.get("raw_visual_similarity"), _safe_float(payload.get("visual_similarity")))
    phash_diff_raw = metadata.get("phash_diff")
    phash_diff = int(phash_diff_raw) if isinstance(phash_diff_raw, (int, float)) else None
    num_sources = int(_safe_float(metadata.get("evidence_sources"), 1))
    num_frames = int(_safe_float(metadata.get("evidence_frames"), 1))
    confidence = _result_confidence(item_type, raw_similarity, num_sources, num_frames, phash_diff)
    if item_type in {"exact_match", "visually_similar"} and raw_similarity > 0:
        payload["visual_similarity"] = raw_similarity
    elif item_type in {"related_content", "embedding_similar"}:
        payload["visual_similarity"] = None
    payload["type"] = item_type
    payload["confidence"] = confidence
    payload["note"] = _result_note(item_type, confidence)
    payload["confidence_level"] = confidence
    payload["match_reason"] = str(metadata.get("match_reason") or payload["note"])
    resolver_status = _resolver_status(item)
    payload["resolver_status"] = resolver_status
    metadata["resolver_status"] = resolver_status
    score_components = dict(metadata.get("score_components") or {})
    payload["score_breakdown"] = {
        "similarity": _safe_float(score_components.get("similarity"), _safe_float(payload.get("visual_similarity"), _safe_float(payload.get("fused_similarity")))),
        "trust": _safe_float(score_components.get("source_trust"), _safe_float(payload.get("credibility_score"), 0.5)),
        "evidence": _safe_float(score_components.get("evidence"), 0.0),
        "final_score": _safe_float(score_components.get("final_score"), _safe_float(metadata.get("final_score"), _safe_float(payload.get("fused_similarity")))),
    }
    metadata["phash_diff"] = phash_diff
    payload["metadata"] = metadata
    return payload


def _timestamp_key(item: dict[str, object]) -> tuple[int, str]:
    timestamp = str(item.get("timestamp") or "").strip()
    return (0, timestamp) if timestamp else (1, "")


def _spread_analysis(exact_matches: list[dict[str, object]]) -> dict[str, object]:
    domains = {
        urlparse(str(item.get("url") or "")).netloc.lower().removeprefix("www.")
        for item in exact_matches
        if item.get("url")
    }
    domains.discard("")
    spread_score = min(1.0, len(exact_matches) / 10.0)
    if spread_score >= 0.7:
        risk_level = "HIGH"
    elif spread_score >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    return {
        "num_exact_matches": len(exact_matches),
        "num_domains": len(domains),
        "spread_score": round(spread_score, 2),
        "risk_level": risk_level,
    }


def _classify_discovery_results(items: list) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    exact_matches: list[dict[str, object]] = []

    for item in items:
        payload = _item_payload(item)
        metadata = dict(payload.get("metadata") or {})
        raw_similarity = _safe_float(metadata.get("raw_visual_similarity"), _safe_float(payload.get("visual_similarity")))
        phash_diff_raw = metadata.get("phash_diff")
        phash_diff = int(phash_diff_raw) if isinstance(phash_diff_raw, (int, float)) else None
        num_sources = int(_safe_float(metadata.get("evidence_sources"), 1))
        is_generic = _is_generic_page(payload)

        if (
            raw_similarity >= EXACT_VISUAL_THRESHOLD
            and phash_diff is not None
            and phash_diff <= EXACT_PHASH_THRESHOLD
            and not is_generic
            and num_sources >= 1
            and _resolver_status(item) == "ok"
            and _is_media_backed_visual_candidate(item)
        ):
            serialized = _serialize_discovery_item(item, "exact_match")
            exact_matches.append(serialized)

    exact_matches.sort(key=lambda entry: _safe_float((entry.get("metadata") or {}).get("final_score"), _safe_float(entry.get("visual_similarity"))), reverse=True)
    return exact_matches, []


def _embedding_similarity_list(
    items: list,
    excluded_keys: set[str] | None = None,
    limit: int = DISCOVERY_SECTION_LIMIT,
) -> list[dict[str, object]]:
    embedding_similar: list[dict[str, object]] = []
    seen_keys: set[str] = set()
    excluded_keys = excluded_keys or set()

    for item in items:
        payload = _item_payload(item)
        if str(payload.get("source_type") or "").lower() != "external":
            continue
        if _is_media_backed_visual_candidate(item):
            continue

        key = _item_identity(item)
        if key and key in excluded_keys:
            continue
        if key and key in seen_keys:
            continue

        serialized = _serialize_discovery_item(item, "embedding_similar")
        metadata = dict(serialized.get("metadata") or {})
        metadata["embedding_rank_score"] = _safe_float(metadata.get("final_score"), _safe_float(serialized.get("fused_similarity")))
        serialized["metadata"] = metadata
        embedding_similar.append(serialized)
        if key:
            seen_keys.add(key)
        if len(embedding_similar) >= limit:
            break

    embedding_similar.sort(
        key=lambda entry: _safe_float((entry.get("metadata") or {}).get("embedding_rank_score"), _safe_float(entry.get("fused_similarity"))),
        reverse=True,
    )
    return embedding_similar[:limit]


def _internal_embedding_similarity_list(
    items: list,
    visual_urls: set[str] | None = None,
    limit: int = DISCOVERY_SECTION_LIMIT,
) -> list[dict[str, object]]:
    embedding_similar: list[dict[str, object]] = []
    visual_urls = visual_urls or set()

    for item in items:
        payload = _item_payload(item)
        if str(payload.get("source_type") or "").lower() == "external":
            continue

        visual_similarity = _safe_float(payload.get("visual_similarity"))
        fused_similarity = _safe_float(payload.get("fused_similarity"))
        if visual_similarity >= INTERNAL_EMBEDDING_MAX_VISUAL_SIMILARITY or fused_similarity <= INTERNAL_EMBEDDING_MIN_SCORE:
            continue

        item_url = str(payload.get("url") or "").strip()
        if item_url and item_url in visual_urls:
            continue

        serialized = _serialize_discovery_item(item, "embedding_similar")
        serialized["note"] = "Semantic neighbor from the internal FAISS index. Useful for broader similarity and investigation."
        metadata = dict(serialized.get("metadata") or {})
        metadata["embedding_rank_score"] = _safe_float(serialized.get("fused_similarity"))
        metadata["semantic_source"] = "internal_faiss"
        serialized["metadata"] = metadata
        embedding_similar.append(serialized)
        if len(embedding_similar) >= limit:
            break

    embedding_similar.sort(
        key=lambda entry: _safe_float((entry.get("metadata") or {}).get("embedding_rank_score"), _safe_float(entry.get("fused_similarity"))),
        reverse=True,
    )
    return embedding_similar[:limit]


def _semantic_embedding_results(
    external_items: list,
    internal_items: list,
    visual_urls: set[str] | None = None,
    limit: int = DISCOVERY_SECTION_LIMIT,
) -> list[dict[str, object]]:
    visual_urls = visual_urls or set()
    external_semantic = _embedding_similarity_list(external_items, excluded_keys=set(), limit=limit)
    external_semantic = [
        item for item in external_semantic
        if not str(item.get("url") or "").strip() or str(item.get("url") or "").strip() not in visual_urls
    ]

    internal_semantic = _internal_embedding_similarity_list(internal_items, visual_urls=visual_urls, limit=limit)

    merged = [*external_semantic, *internal_semantic]
    merged.sort(
        key=lambda entry: _safe_float((entry.get("metadata") or {}).get("embedding_rank_score"), _safe_float(entry.get("fused_similarity"))),
        reverse=True,
    )
    return merged[:limit]


def _merge_external_discovery_items(*groups: list) -> list:
    merged: list = []
    seen_keys: set[str] = set()

    for group in groups:
        for item in group:
            key = _item_identity(item)
            if key and key in seen_keys:
                continue
            merged.append(item)
            if key:
                seen_keys.add(key)
    return merged


def _visual_similarity_list(items: list, excluded_keys: set[str] | None = None, limit: int = DISCOVERY_SECTION_LIMIT) -> list[dict[str, object]]:
    visually_similar: list[dict[str, object]] = []
    excluded_keys = excluded_keys or set()
    for item in items:
        key = _item_identity(item)
        if key and key in excluded_keys:
            continue
        payload = _item_payload(item)
        metadata = dict(payload.get("metadata") or {})
        raw_similarity = _safe_float(metadata.get("raw_visual_similarity"), _safe_float(payload.get("visual_similarity")))
        if raw_similarity < RELATED_VISUAL_THRESHOLD:
            continue
        if _resolver_status(item) != "ok":
            continue
        if not _is_media_backed_visual_candidate(item):
            continue
        payload = _serialize_discovery_item(item, "visually_similar")
        payload["metadata"] = {
            **dict(payload.get("metadata") or {}),
            "visual_rank_score": _safe_float(payload.get("visual_similarity")),
        }
        visually_similar.append(payload)

    visually_similar.sort(
        key=lambda entry: (
            _safe_float(entry.get("visual_similarity")),
            _safe_float((entry.get("metadata") or {}).get("final_score")),
        ),
        reverse=True,
    )
    deduped: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    for item in visually_similar:
        key = _item_identity(item)
        if key and key in seen_urls:
            continue
        if key:
            seen_urls.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _embedding_matches_top10(items: list, excluded_keys: set[str] | None = None, limit: int = DISCOVERY_SECTION_LIMIT) -> list[dict[str, object]]:
    embedding_candidates: list[dict[str, object]] = []
    excluded_keys = excluded_keys or set()
    seen_keys: set[str] = set()

    for item in items:
        key = _item_identity(item)
        if key and key in excluded_keys:
            continue
        if key and key in seen_keys:
            continue

        embedding_similarity = _embedding_similarity_value(item)
        if embedding_similarity < EMBEDDING_MATCH_THRESHOLD:
            continue

        serialized = _serialize_discovery_item(item, "embedding_similar")
        metadata = dict(serialized.get("metadata") or {})
        metadata["embedding_similarity"] = embedding_similarity
        metadata["embedding_rank_score"] = embedding_similarity
        serialized["metadata"] = metadata
        embedding_candidates.append(serialized)
        if key:
            seen_keys.add(key)

    embedding_candidates.sort(
        key=lambda entry: (
            _safe_float((entry.get("metadata") or {}).get("embedding_similarity")),
            _safe_float((entry.get("metadata") or {}).get("final_score"), _safe_float(entry.get("fused_similarity"))),
        ),
        reverse=True,
    )
    return embedding_candidates[:limit]


def _build_external_exploratory_items(
    pipeline,
    stored_path: Path,
    source_url: str | None,
    limit: int = DISCOVERY_SECTION_LIMIT,
) -> list[dict[str, object]]:
    from ai.layer2_matching.audio.audio_extract import extract_audio_from_media
    from ai.layer2_matching.tracking.metadata_parser import OccurrenceRecord, credibility_score_for_source, infer_platform, normalize_timestamp

    external_client = pipeline.external_search
    visual_embedding = pipeline.visual_embedder.embed_media(stored_path)
    audio_extraction = extract_audio_from_media(stored_path, pipeline.config.cache_dir / "uploads")
    audio_embedding = pipeline.audio_embedder.embed_audio(
        waveform=audio_extraction.waveform,
        sample_rate=audio_extraction.sample_rate,
        duration_seconds=audio_extraction.duration_seconds,
    )

    reverse_candidates = []
    public_source_url = external_client._public_source_url(source_url)
    if public_source_url:
        requested_provider_results = max(limit * 2, DISCOVERY_SECTION_LIMIT * 2)
        for provider in external_client.reverse_providers:
            cached = external_client._load_provider_cache(provider.name, stored_path)
            provider_candidates = cached if cached is not None and len(cached) >= requested_provider_results else []
            if cached is None or len(provider_candidates) < requested_provider_results:
                try:
                    provider_candidates = provider.search_image(public_source_url, max_results=requested_provider_results)
                except Exception:
                    provider_candidates = []
                provider_candidates = external_client._sanitize_provider_candidates(provider_candidates, public_source_url)
                if provider_candidates:
                    external_client._save_provider_cache(provider.name, stored_path, provider_candidates)
            else:
                provider_candidates = external_client._sanitize_provider_candidates(provider_candidates, public_source_url)
            reverse_candidates.extend(provider_candidates)
    else:
        query_images = external_client._prepare_query_images(stored_path)
        for frame_index, image_path in enumerate(query_images, start=1):
            reverse_candidates.extend(
                external_client._reverse_search_frame(image_path, frame_index=frame_index, max_results=limit * 2)
            )

    merged_candidates = external_client._merge_reverse_candidates(reverse_candidates)
    exploratory_records = []
    seen_keys: set[str] = set()

    for candidate in merged_candidates:
        raw_candidates = list(candidate.get("raw_candidates") or [])
        provider_search_types = sorted(
            {
                str((raw_candidate.metadata or {}).get("provider_search_type") or "").strip()
                for raw_candidate in raw_candidates
                if str((raw_candidate.metadata or {}).get("provider_search_type") or "").strip()
            }
        )
        exact_matches_hint = any(bool((raw_candidate.metadata or {}).get("exact_matches_hint")) for raw_candidate in raw_candidates)
        fallback_media_urls: list[str] = []
        for raw_candidate in raw_candidates:
            raw_metadata = dict(raw_candidate.metadata or {})
            for candidate_url in (
                raw_candidate.media_url,
                raw_metadata.get("thumbnail"),
                raw_metadata.get("provider_image_url"),
                raw_metadata.get("provider_media_url"),
            ):
                cleaned_url = str(candidate_url or "").strip()
                if cleaned_url and cleaned_url not in fallback_media_urls:
                    fallback_media_urls.append(cleaned_url)

        resolution_targets: list[str | None] = []
        primary_media_url = str(candidate.get("media_url") or "").strip() or None
        if primary_media_url:
            resolution_targets.append(primary_media_url)
        for fallback_url in fallback_media_urls:
            if fallback_url not in resolution_targets:
                resolution_targets.append(fallback_url)
        if not resolution_targets:
            resolution_targets.append(None)

        resolved = None
        try:
            for media_url_candidate in resolution_targets:
                resolved_attempt = external_client.resolver.resolve(
                    page_url=str(candidate.get("page_url") or "") or None,
                    media_url=media_url_candidate,
                    title=str(candidate.get("title") or "") or None,
                    caption=str(candidate.get("snippet") or "") or None,
                    timestamp=str(candidate.get("timestamp") or "") or None,
                    metadata={"reverse_providers": list(candidate.get("providers") or [])},
                )
                if resolved_attempt is None:
                    continue
                resolved = resolved_attempt
                if resolved_attempt.local_path is not None and resolved_attempt.local_path.exists():
                    break
        except Exception:
            resolved = None

        verification = None
        if resolved is not None and resolved.local_path is not None and resolved.local_path.exists():
            try:
                verification = external_client.verifier.verify_candidate(
                    candidate_path=resolved.local_path,
                    original_visual_embedding=visual_embedding,
                    original_audio_embedding=audio_embedding.combined_embedding,
                    original_media_path=stored_path,
                )
            except Exception:
                verification = None

        page_url = str(candidate.get("page_url") or (resolved.page_url if resolved else "") or "").strip() or None
        if external_client._is_low_signal_url(page_url):
            continue
        key = page_url or str(candidate.get("candidate_key") or "")
        if key and key in seen_keys:
            continue

        platform = infer_platform(page_url, fallback=resolved.platform if resolved else "external")
        best_rank = int(candidate.get("best_rank") or 9999)
        provider_hits = int(candidate.get("provider_hits") or 0)
        frame_hits = int(candidate.get("frame_hits") or 0)
        base_rank_score = max(0.08, 0.82 - 0.06 * max(best_rank - 1, 0))
        visual_similarity = verification.visual_similarity if verification is not None else None
        audio_similarity = verification.audio_similarity if verification is not None else None
        fused_similarity = verification.combined_score if verification is not None and verification.combined_score > 0 else base_rank_score
        caption = (
            (resolved.caption if resolved and resolved.caption else None)
            or str(candidate.get("snippet") or "").strip()
            or None
        )
        title = (
            (resolved.title if resolved and resolved.title else None)
            or str(candidate.get("title") or "").strip()
            or page_url
        )
        metadata = {
            "provider": "reverse_search",
            "reverse_providers": list(candidate.get("providers") or []),
            "provider_search_types": provider_search_types,
            "provider_hits": provider_hits,
            "frame_hits": frame_hits,
            "best_provider_rank": best_rank,
            "media_url": resolved.media_url if resolved else str(candidate.get("media_url") or "") or None,
            "resolved_media_type": resolved.media_type if resolved else None,
            "downloaded_path": str(resolved.local_path) if resolved and resolved.local_path else None,
            "match_reason": (
                "Ranked from reverse-image candidate pages because strict verification returned limited exact matches."
            ),
            "candidate_key": str(candidate.get("candidate_key") or ""),
            "resolved_media_url": (resolved.metadata.get("resolved_media_url") if resolved else None),
            "resolved_image_urls": list((resolved.metadata.get("resolved_image_urls") if resolved else []) or []),
            "resolved_video_urls": list((resolved.metadata.get("resolved_video_urls") if resolved else []) or []),
            "resolved_audio_urls": list((resolved.metadata.get("resolved_audio_urls") if resolved else []) or []),
            "raw_visual_similarity": visual_similarity,
            "provider_visual_rank_score": base_rank_score if "google_lens_visual_matches" in provider_search_types else None,
            "exact_matches_hint": exact_matches_hint,
            "visual_threshold": getattr(external_client.verifier, "visual_threshold", None),
            "audio_threshold": getattr(external_client.verifier, "audio_threshold", None),
            "phash_diff": verification.metadata.get("phash_diff") if verification is not None else None,
            "evidence_sources": provider_hits,
            "evidence_frames": frame_hits,
            **({"source_url": source_url} if source_url else {}),
        }
        record = OccurrenceRecord(
            entry_id=str(candidate.get("candidate_key") or key or f"candidate-{best_rank}"),
            source_type="external",
            platform=platform,
            url=page_url,
            local_path=None,
            timestamp=normalize_timestamp(resolved.timestamp if resolved else candidate.get("timestamp")),
            title=title,
            caption=caption,
            label=None,
            credibility_score=credibility_score_for_source(platform, page_url),
            visual_similarity=visual_similarity,
            audio_similarity=audio_similarity,
            fused_similarity=float(fused_similarity),
            context="news",
            context_scores={"news": 0.34, "meme": 0.33, "propaganda / misinformation": 0.33},
            is_mock=False,
            metadata=metadata,
        )
        adjusted = external_client._adjust_record_score(record)
        item = {
            "id": adjusted.entry_id,
            "url": adjusted.url,
            "media_url": str(adjusted.metadata.get("media_url")) if adjusted.metadata.get("media_url") else None,
            "local_path": adjusted.local_path,
            "visual_similarity": adjusted.visual_similarity,
            "audio_similarity": adjusted.audio_similarity,
            "fused_similarity": adjusted.fused_similarity,
            "combined_score": adjusted.fused_similarity,
            "platform": adjusted.platform,
            "timestamp": adjusted.timestamp,
            "source_type": adjusted.source_type,
            "title": adjusted.title,
            "caption": adjusted.caption,
            "context": adjusted.context,
            "context_scores": adjusted.context_scores,
            "credibility_score": adjusted.credibility_score,
            "is_mock": adjusted.is_mock,
            "label": adjusted.label,
            "metadata": adjusted.metadata,
        }
        exploratory_records.append(item)
        if key:
            seen_keys.add(key)
        if len(exploratory_records) >= limit * 3:
            break

    return exploratory_records


def _get_layer2_pipeline():
    global _LAYER2_PIPELINE
    if _LAYER2_PIPELINE is None:
        from ai.layer2_matching.pipeline import Layer2Pipeline

        _LAYER2_PIPELINE = Layer2Pipeline(PROJECT_ROOT)
    return _LAYER2_PIPELINE


def _run_layer2_discovery(
    *,
    upload_id: str,
    stored_path: Path,
    upload_metadata: dict[str, object],
) -> dict[str, object]:
    errors: list[str] = []
    pipeline = _get_layer2_pipeline()

    public_source_url: str | None
    try:
        public_source_url = _ensure_public_source_url(upload_id, upload_metadata)
    except Exception as exc:  # pragma: no cover - network/provider safety
        LOGGER.exception("Failed to establish a public source URL for %s", stored_path)
        public_source_url = None
        errors.append(f"Cloudinary upload failed: {exc}")

    original_source_url = str(upload_metadata.get("source_url") or "").strip() or None
    layer2_source_url = public_source_url or original_source_url

    manual_search_links, manual_search_note = _manual_search_links(
        str(upload_metadata.get("original_filename") or stored_path.name),
        source_url=layer2_source_url,
    )

    external_items: list[dict[str, object]] = []
    reverse_search_payload = _empty_reverse_search_payload()
    reverse_confidence_score = 0.0
    fallback_used = False
    serpapi_status = "failed"

    if layer2_source_url:
        try:
            from ai.layer2_matching.tracking.reverse_search_service import process_reverse_search

            reverse_result = process_reverse_search(image_url=layer2_source_url)
            reverse_search_payload = dict(reverse_result.get("reverse_search") or _empty_reverse_search_payload())
            reverse_confidence_score = float(reverse_result.get("confidence_score") or 0.0)
            fallback_used = bool(reverse_result.get("fallback_used"))
            serpapi_status = "ok"
        except Exception as exc:  # pragma: no cover - network/provider safety
            LOGGER.exception("Public reverse search failed for %s", stored_path)
            errors.append(f"Public reverse search failed: {exc}")

    try:
        external_items = _build_external_exploratory_items(
            pipeline,
            stored_path,
            layer2_source_url,
            limit=DISCOVERY_SECTION_LIMIT,
        )
    except Exception as exc:  # pragma: no cover - network/provider safety
        LOGGER.exception("External Layer 2 discovery failed for %s", stored_path)
        errors.append(f"External reverse search failed: {exc}")

    exact_matches, _ = _classify_discovery_results(external_items)
    exact_matches = exact_matches[:DISCOVERY_SECTION_LIMIT]
    exact_keys = {identity for item in exact_matches if (identity := _item_identity(item))}
    verified_visual_matches = _visual_similarity_list(
        external_items,
        excluded_keys=exact_keys,
        limit=DISCOVERY_SECTION_LIMIT,
    )
    exact_urls = {
        str(item.get("page_url") or item.get("url") or "").strip()
        for item in exact_matches
        if str(item.get("page_url") or item.get("url") or "").strip()
    }
    raw_visual_matches = _reverse_visual_matches(
        reverse_search_payload,
        excluded_urls=exact_urls,
        limit=DISCOVERY_SECTION_LIMIT,
    )
    visual_matches_top10 = _merge_section_items(
        verified_visual_matches,
        raw_visual_matches,
        limit=DISCOVERY_SECTION_LIMIT,
    )
    visual_urls = {
        str(item.get("page_url") or item.get("url") or "").strip()
        for item in [*exact_matches, *visual_matches_top10]
        if str(item.get("page_url") or item.get("url") or "").strip()
    }
    embedding_matches_top10 = _related_web_source_matches(
        reverse_search_payload,
        excluded_urls=visual_urls,
        limit=DISCOVERY_SECTION_LIMIT,
    )

    has_matches = bool(exact_matches or visual_matches_top10 or embedding_matches_top10)
    return build_layer2_response(
        {
            "exact_matches": exact_matches,
            "visual_matches_top10": visual_matches_top10,
            "embedding_matches_top10": embedding_matches_top10,
            "related_web_sources": embedding_matches_top10,
            "spread_analysis": _spread_analysis(exact_matches),
            "provider_status": {
                "cloudinary": "ok" if layer2_source_url else "failed",
                "serpapi": serpapi_status,
            },
            "image_url": layer2_source_url or "",
            "reverse_search": reverse_search_payload,
            "sources": list(reverse_search_payload.get("sources") or []),
            "manual_search_links": manual_search_links,
            "manual_search_note": manual_search_note,
            "fallback_used": fallback_used,
            "confidence_score": reverse_confidence_score,
            "errors": errors,
            "message": "Matches found" if has_matches else "No matches found",
            "status": "success" if not errors else "degraded",
        }
    )


def _request_image_url() -> str | None:
    payload = request.get_json(silent=True) if request.is_json else {}
    image_url = ""
    if isinstance(payload, dict):
        image_url = str(payload.get("image_url") or "").strip()
    if not image_url:
        image_url = str(request.form.get("image_url") or "").strip()
    return image_url or None


def _build_tracking_timeline(*, confidence: float, is_fake: bool) -> list[dict[str, object]]:
    base = max(3, int(round((confidence * 100) / 14)))
    multipliers = [0.7, 0.82, 0.95, 1.05, 1.18, 1.26, 1.45, 1.63, 1.8, 1.95, 2.15, 2.32]
    now_utc = datetime.now(timezone.utc)
    timeline: list[dict[str, object]] = []
    for index, factor in enumerate(multipliers):
        spread_bias = 1.25 if is_fake else 0.75
        mentions = max(1, int(round(base * factor * spread_bias)))
        point_time = now_utc.replace(minute=0, second=0, microsecond=0) - timedelta(hours=(len(multipliers) - 1 - index))
        timeline.append(
            {
                "timestamp": point_time.isoformat(),
                "mentions": mentions,
            }
        )
    return timeline


def _build_dashboard_risk(*, confidence: float, timeline: list[dict[str, object]], source_credibility: float) -> dict[str, float]:
    fake_probability = max(0.0, min(1.0, confidence))
    if len(timeline) >= 2:
        first = float(timeline[0]["mentions"])
        last = float(timeline[-1]["mentions"])
        spread_velocity = max(0.0, min(1.0, (last - first) / max(1.0, last)))
    else:
        spread_velocity = 0.0
    credibility = max(0.0, min(1.0, source_credibility))
    risk_score = max(0.0, min(1.0, (0.5 * fake_probability) + (0.3 * spread_velocity) + (0.2 * (1.0 - credibility))))
    return {
        "risk_score": round(risk_score, 4),
        "fake_probability": round(fake_probability, 4),
        "spread_velocity": round(spread_velocity, 4),
        "source_credibility": round(credibility, 4),
    }


def _allowed_cors_origins() -> set[str]:
    allowed = {origin for origin in CORS_ALLOWED_ORIGINS if origin}
    if DASHBOARD_FRONTEND_URL:
        parsed = urlparse(DASHBOARD_FRONTEND_URL)
        if parsed.scheme and parsed.netloc:
            allowed.add(f"{parsed.scheme}://{parsed.netloc}")
    return allowed


def _is_json_api_request() -> bool:
    path = str(request.path or "")
    return path.startswith("/api/") or path in {"/reverse-search", "/health"}


def _inject_response_request_id(response):
    request_id = _current_request_id()
    response.headers["X-Request-ID"] = request_id
    if response.is_json:
        payload = response.get_json(silent=True)
        if isinstance(payload, dict):
            payload.setdefault("request_id", request_id)
            if "status" not in payload and response.status_code >= 400:
                payload["status"] = "error"
                payload.setdefault("message", str(payload.get("error") or "Request failed"))
            response.set_data(json.dumps(payload))
            response.mimetype = "application/json"
    return response


@app.before_request
def initialize_request_context():
    g.request_id = _new_uuid()
    g.request_started_at = time.time()
    _cleanup_layer2_store()


@app.after_request
def disable_cache(response):
    started_at = float(getattr(g, "request_started_at", 0.0) or 0.0)
    elapsed = time.time() - started_at if started_at > 0 else 0.0
    if elapsed > REQUEST_TIMEOUT_SECONDS and _is_json_api_request():
        response.headers["X-Slow-Request"] = f"{elapsed:.2f}s"

    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    origin = str(request.headers.get("Origin") or "").strip()
    allowed = _allowed_cors_origins()
    if origin and origin in allowed:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Vary"] = "Origin"
    return _inject_response_request_id(response)


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def api_preflight(_any: str):
    return make_response("", 204)


@app.get("/")
def index():
    return render_template("index.html", **_template_context(service_entry=False))


@app.get("/minimal")
def minimal_index():
    return render_template("index.html", **_template_context(service_entry=False))


@app.get("/service")
@jwt_required(AUTH_SERVICE, allow_guest=True, redirect_on_fail=True)
def service():
    return redirect(url_for("dashboard"))


@app.get("/dashboard")
@jwt_required(AUTH_SERVICE, allow_guest=True, redirect_on_fail=True)
def dashboard():
    if DASHBOARD_FRONTEND_URL:
        return redirect(f"{DASHBOARD_FRONTEND_URL.rstrip('/')}/dashboard")
    return render_template(
        "dashboard.html",
        **_template_context(service_entry=True),
        dashboard_frontend_url=DASHBOARD_FRONTEND_URL,
    )


@app.get("/demo")
def demo():
    return redirect(url_for("dashboard"))


def _ping_service(url: str, *, timeout_seconds: float = 2.0) -> str:
    if not url:
        return "unknown"
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(url, timeout=timeout_seconds, verify=False, allow_redirects=True)
        return "ok" if int(response.status_code) < 500 else "failed"
    except Exception:
        return "failed"
    finally:
        session.close()


def _health_service_statuses() -> dict[str, str]:
    cloudinary_configured = bool(str(os.getenv("CLOUDINARY_URL") or "").strip())
    serpapi_configured = bool(str(os.getenv("SERPAPI_API_KEY") or "").strip())
    proxy_configured = bool(REMOTE_PROXY_URL)

    cloudinary_status = _ping_service("https://api.cloudinary.com") if cloudinary_configured else "unknown"
    serpapi_status = _ping_service("https://serpapi.com") if serpapi_configured else "unknown"
    proxy_status = _ping_service(f"{REMOTE_PROXY_URL.rstrip('/')}/health") if proxy_configured else "unknown"

    if proxy_configured and proxy_status == "failed":
        proxy_status = _ping_service(REMOTE_PROXY_URL.rstrip("/"))

    return {
        "cloudinary": cloudinary_status,
        "serpapi": serpapi_status,
        "proxy": proxy_status,
    }


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "services": _health_service_statuses(),
            "build": APP_BUILD,
            "missing_models": _missing_models(),
        }
    )


@app.post("/api/analyze")
@jwt_required(AUTH_SERVICE, allow_guest=True)
def api_analyze():
    principal = dict(getattr(g, "auth_principal", {}) or {})
    guest_allowed, guest_error, guest_usage_snapshot = AUTH_SERVICE.enforce_guest_quota(principal)
    if not guest_allowed:
        return jsonify({"error": guest_error, "auth_state": principal.get("auth_mode"), **guest_usage_snapshot}), 403

    missing_models = _missing_models()
    if missing_models:
        return jsonify({"error": "Required model files are missing.", "missing_models": missing_models}), 503

    analyze_request = parse_analyze_request(dict(request.form))
    source_url = analyze_request.source_url
    uploaded_file = request.files.get("file")
    has_uploaded_file = uploaded_file is not None and bool(uploaded_file.filename)
    if not has_uploaded_file and not source_url:
        return jsonify({"error": "Upload an image/video or provide a source URL before analysis."}), 400

    if has_uploaded_file:
        suffix = Path(uploaded_file.filename).suffix.lower() or ".jpg"
        if suffix not in ALLOWED_INPUT_SUFFIXES:
            return jsonify({"error": "Unsupported file type. Use JPG, JPEG, PNG, or supported video formats."}), 400

    enable_layer1 = analyze_request.enable_layer1
    enable_layer2 = analyze_request.enable_layer2
    enable_layer3 = analyze_request.enable_layer3
    guest_usage = AUTH_SERVICE.mark_guest_try_used(principal)

    try:
        if has_uploaded_file:
            upload_id, stored_path = _save_uploaded_file(uploaded_file)
            original_filename = str(uploaded_file.filename or stored_path.name)
        else:
            upload_id, stored_path, original_filename = _download_remote_media(source_url or "")
        label, confidence = _run_inference_subprocess(stored_path)
        _store_upload_metadata(
            upload_id=upload_id,
            stored_path=stored_path,
            original_filename=original_filename,
            label=label,
            confidence=confidence,
            source_url=source_url,
        )
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc), "auth_state": principal.get("auth_mode"), "guest_usage": guest_usage}), 400
    except Exception as exc:  # pragma: no cover - route safety
        message = str(exc)
        lowered = message.lower()
        if any(token in lowered for token in ("cannot identify image", "invalid image", "unidentifiedimageerror", "unsupported")):
            return jsonify({"error": "Invalid or unsupported media file.", "auth_state": principal.get("auth_mode"), "guest_usage": guest_usage}), 400
        return jsonify({"error": f"Analysis failed: {message}", "auth_state": principal.get("auth_mode"), "guest_usage": guest_usage}), 500

    now_iso = datetime.now(timezone.utc).isoformat()
    is_fake = label.lower() == "fake"

    layer1_payload: dict[str, object] = build_layer1_payload(is_fake=is_fake, confidence=confidence)
    if not enable_layer1:
        layer1_payload = {"result": "REAL", "confidence": 0.0, "heatmap": None}

    layer2_payload = build_layer2_response(None)
    upload_metadata = _load_upload_metadata(upload_id)
    if enable_layer2:
        layer2_payload = _run_layer2_discovery(
            upload_id=upload_id,
            stored_path=stored_path,
            upload_metadata=upload_metadata,
        )
    consistency_warning = _consistency_warning_payload(
        str(layer1_payload.get("result") or ""),
        float(layer1_payload.get("confidence") or 0.0),
        list(layer2_payload.get("exact_matches") or []),
    )
    layer2_payload["consistency_warning"] = consistency_warning
    _store_layer2_channels(upload_id, layer2_payload)

    timeline: list[dict[str, object]] = []
    alerts: list[dict[str, object]] = []
    growth_rate = 0.0
    risk_metrics = {
        "risk_score": 0.0,
        "fake_probability": 0.0,
        "spread_velocity": 0.0,
        "source_credibility": 0.0,
    }
    growth_payload: dict[str, object] = {"rate_percent": 0.0, "spike_detected": False, "window": "1h"}
    risk_score = 0.0
    growth_indicator = "low"
    risk_insight = "No spread data yet. Run analysis again after new activity appears."

    if enable_layer3:
        source_credibility = 0.42 if is_fake else 0.78
        timeline = _build_tracking_timeline(confidence=confidence, is_fake=is_fake)
        risk_metrics = _build_dashboard_risk(
            confidence=confidence,
            timeline=timeline,
            source_credibility=source_credibility,
        )
        if len(timeline) >= 2:
            first_mentions = float(timeline[0]["mentions"])
            last_mentions = float(timeline[-1]["mentions"])
            growth_rate = round(((last_mentions - first_mentions) / max(1.0, first_mentions)) * 100.0, 2)
        spike_detected = growth_rate >= 120
        risk_score = float(risk_metrics.get("risk_score") or 0.0)
        growth_payload = {"rate_percent": growth_rate, "spike_detected": spike_detected, "window": "1h"}
        growth_indicator = build_growth_indicator(growth_rate)
        risk_insight = build_risk_insight(
            fake_probability=confidence if is_fake else (1.0 - confidence),
            growth_rate_percent=growth_rate,
            source_count=int(layer2_payload["counts"]["exact"]),
        )
        alerts = [
            {
                "id": f"alert-{upload_id}-{index}",
                **alert,
            }
            for index, alert in enumerate(
                build_alerts(
                    growth_rate_percent=growth_rate,
                    source_count=int(layer2_payload["counts"]["exact"]),
                ),
                start=1,
            )
        ]

    return jsonify(
        {
            "analysis_id": upload_id,
            "upload_id": upload_id,
            "auth_state": principal.get("auth_mode", "anonymous"),
            "guest_usage": guest_usage,
            "layer1": layer1_payload,
            "layer2": layer2_payload,
            "origin_summary": layer2_payload.get("origin_summary"),
            "top_domains": layer2_payload.get("top_domains"),
            "first_seen_estimate": layer2_payload.get("first_seen_estimate"),
            "domain_clusters": layer2_payload.get("domain_clusters"),
            "consistency_warning": consistency_warning,
            "layer3": {
                "timeline": timeline,
                "growth": growth_payload,
                "growth_rate": growth_rate,
                "growth_indicator": growth_indicator,
                "alerts": alerts,
                "risk_score": risk_score,
                "risk": risk_metrics,
                "risk_insight": risk_insight,
            },
            "meta": {
                "upload_id": upload_id,
                "filename": Path(original_filename).name,
                "created_at": now_iso,
                "model_version": "fusion-v1",
            },
        }
    )


@app.post("/api/chat")
@jwt_required(AUTH_SERVICE, allow_guest=True)
def api_chat():
    payload = request.get_json(silent=True) or {}
    try:
        parsed_request = parse_chat_request(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    message = parsed_request.message
    layer1 = parsed_request.layer1
    layer2 = parsed_request.layer2
    layer3 = parsed_request.layer3

    l1_result = str(layer1.get("result") or "UNKNOWN").upper()
    l1_conf = float(layer1.get("confidence") or 0.0)
    matches = layer2.get("matches") if isinstance(layer2.get("matches"), list) else []
    timeline = layer3.get("timeline") if isinstance(layer3.get("timeline"), list) else []
    growth_obj = layer3.get("growth") if isinstance(layer3.get("growth"), dict) else {}
    growth_rate = float(growth_obj.get("rate_percent") or layer3.get("growth_rate") or 0.0)
    risk_score = float(layer3.get("risk_score") or 0.0)
    alerts = layer3.get("alerts") if isinstance(layer3.get("alerts"), list) else []

    context_ready = bool(layer1 or layer2 or layer3)
    if not context_ready:
        reply = (
            "I can help explain results once an analysis is run. Upload media first, then ask about authenticity, "
            "sources, spread, or risk."
        )
    else:
        verdict_line = f"Layer 1 verdict is {l1_result} with {round(l1_conf, 2)}% confidence."
        source_line = f"Layer 2 found {len(matches)} source match(es)."
        track_line = f"Layer 3 growth is {round(growth_rate, 2)}% with risk score {round(risk_score * 100, 1)}%."
        alert_line = f"There are {len(alerts)} active alert(s)."
        response_focus = "Focus next on manual verification and source triage." if l1_result == "FAKE" else "Focus next on corroboration and monitoring drift."
        reply = f"{verdict_line} {source_line} {track_line} {alert_line} {response_focus}"

    return jsonify(
        {
            "reply": reply,
            "context_used": context_ready,
            "analysis_summary": {
                "layer1_result": l1_result,
                "layer1_confidence": round(l1_conf, 2),
                "match_count": len(matches),
                "timeline_points": len(timeline),
                "growth_rate_percent": round(growth_rate, 2),
                "risk_score_percent": round(risk_score * 100, 2),
                "alert_count": len(alerts),
            },
        }
    )


@app.post("/api/predict")
@jwt_required(AUTH_SERVICE, allow_guest=True)
def api_predict():
    principal = dict(getattr(g, "auth_principal", {}) or {})

    guest_allowed, guest_error, guest_usage_snapshot = AUTH_SERVICE.enforce_guest_quota(principal)
    if not guest_allowed:
        return jsonify({"error": guest_error, "auth_state": principal.get("auth_mode"), **guest_usage_snapshot}), 403

    missing_models = _missing_models()
    if missing_models:
        return jsonify({"error": "Required model files are missing.", "missing_models": missing_models}), 503

    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Upload an image before running analysis."}), 400

    suffix = Path(uploaded_file.filename).suffix.lower() or ".jpg"
    if suffix not in ALLOWED_INPUT_SUFFIXES:
        return jsonify({"error": "Unsupported file type. Use JPG, JPEG, PNG, or a supported video file."}), 400
    source_url = str(request.form.get("source_url") or "").strip() or None
    guest_usage = AUTH_SERVICE.mark_guest_try_used(principal)

    try:
        upload_id, stored_path = _save_uploaded_file(uploaded_file)
        label, confidence = _run_inference_subprocess(stored_path)
        _store_upload_metadata(
            upload_id=upload_id,
            stored_path=stored_path,
            original_filename=uploaded_file.filename,
            label=label,
            confidence=confidence,
            source_url=source_url,
        )
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc), "auth_state": principal.get("auth_mode"), "guest_usage": guest_usage}), 400
    except Exception as exc:  # pragma: no cover - demo safety
        message = str(exc)
        lowered = message.lower()
        if any(token in lowered for token in ("cannot identify image", "invalid image", "unidentifiedimageerror", "unsupported")):
            return jsonify({"error": "Invalid or unsupported media file.", "auth_state": principal.get("auth_mode"), "guest_usage": guest_usage}), 400
        return jsonify({"error": f"Inference failed: {message}", "auth_state": principal.get("auth_mode"), "guest_usage": guest_usage}), 500

    return jsonify(
        {
            "upload_id": upload_id,
            "label": label.upper(),
            "label_raw": label.lower(),
            "confidence": round(confidence * 100, 2),
            "filename": Path(uploaded_file.filename).name,
            "reasoning": _build_reasoning(label=label, confidence=confidence, original_filename=uploaded_file.filename),
            "next_step": "If you want more context, run the next step to search the internet for similar appearances and ranked source links.",
            "source_url_used": bool(source_url),
            "auth_state": principal.get("auth_mode", "anonymous"),
            "guest_usage": guest_usage,
        }
    )


@app.post("/api/discover")
@jwt_required(AUTH_SERVICE, allow_guest=True)
def api_discover():
    payload = request.get_json(silent=True) or {}
    upload_id = str(payload.get("upload_id") or "").strip()
    if not upload_id:
        return jsonify({"status": "error", "message": "upload_id required"}), 400

    stored_layer2 = _get_layer2_channels(upload_id)
    if stored_layer2 is None:
        try:
            upload_metadata = _load_upload_metadata(upload_id)
        except FileNotFoundError:
            return jsonify({"status": "error", "message": "invalid upload_id"}), 404

        stored_path = Path(str(upload_metadata.get("stored_path") or ""))
        if not stored_path.exists():
            return jsonify({"status": "error", "message": "stored media file is missing"}), 404

        stored_layer2 = _run_layer2_discovery(
            upload_id=upload_id,
            stored_path=stored_path,
            upload_metadata=upload_metadata,
        )
        _store_layer2_channels(upload_id, stored_layer2)

    return jsonify(build_layer2_response(stored_layer2))


def _empty_reverse_search_payload(errors: list[str] | None = None) -> dict[str, object]:
    return {
        "matches_found": False,
        "top_matches": [],
        "similar_images": [],
        "visual_matches": [],
        "exact_matches": [],
        "embedding_matches": [],
        "context_analysis": [],
        "clusters": [],
        "spread_analysis": {"origin": "", "total_sources": 0, "risk_level": "low"},
        "scraped_data": [],
        "image_results": [],
        "inline_images": [],
        "sources": [],
        "errors": errors or [],
    }


def _channels_from_reverse_payload(reverse_search: dict[str, object]) -> tuple[list[dict], list[dict], list[dict], dict[str, int]]:
    exact_matches = list(reverse_search.get("exact_matches") or reverse_search.get("top_matches") or [])[:10]
    visual_matches_top10 = list(reverse_search.get("visual_matches") or [])[:10]
    embedding_matches_top10 = list(reverse_search.get("embedding_matches") or [])[:10]
    counts = {
        "exact": len(exact_matches),
        "visual": len(visual_matches_top10),
        "embedding": len(embedding_matches_top10),
    }
    return exact_matches, visual_matches_top10, embedding_matches_top10, counts


def _normalized_provider_status(raw_providers: dict[str, object] | None) -> dict[str, str]:
    providers = dict(raw_providers or {})
    cloudinary_raw = str(providers.get("cloudinary") or "failed").strip().lower()
    serpapi_raw = str(providers.get("serpapi") or "failed").strip().lower()
    cloudinary = "ok" if cloudinary_raw == "ok" else "failed"
    serpapi = "ok" if serpapi_raw == "ok" else "failed"
    return {"cloudinary": cloudinary, "serpapi": serpapi}


def _contract_reverse_response(
    *,
    execution: str,
    status: str,
    message: str,
    image_url: str = "",
    reverse_search: dict[str, object] | None = None,
    fallback_used: bool = False,
    confidence_score: float = 0.0,
    providers: dict[str, object] | None = None,
) -> dict[str, object]:
    reverse_search_data = dict(reverse_search or _empty_reverse_search_payload())
    reverse_search_data.setdefault("matches_found", False)
    reverse_search_data.setdefault("top_matches", [])
    reverse_search_data.setdefault("similar_images", [])
    reverse_search_data.setdefault("visual_matches", [])
    reverse_search_data.setdefault("exact_matches", [])
    reverse_search_data.setdefault("embedding_matches", [])
    reverse_search_data.setdefault("context_analysis", [])
    reverse_search_data.setdefault("clusters", [])
    reverse_search_data.setdefault("spread_analysis", {"origin": "", "total_sources": 0, "risk_level": "low"})
    reverse_search_data.setdefault("scraped_data", [])
    reverse_search_data.setdefault("image_results", [])
    reverse_search_data.setdefault("inline_images", [])
    reverse_search_data.setdefault("sources", [])
    reverse_search_data.setdefault("errors", [])

    exact_matches, visual_matches_top10, embedding_matches_top10, counts = _channels_from_reverse_payload(reverse_search_data)
    provider_status = _normalized_provider_status(providers)

    response: dict[str, object] = {
        "execution": execution,
        "status": status,
        "message": message,
        "image_url": image_url,
        "reverse_search": reverse_search_data,
        "fallback_used": bool(fallback_used),
        "confidence_score": float(confidence_score),
        "exact_matches": exact_matches,
        "visual_matches": list(reverse_search_data.get("visual_matches") or visual_matches_top10),
        "embedding_matches": list(reverse_search_data.get("embedding_matches") or embedding_matches_top10),
        "visual_matches_top10": visual_matches_top10,
        "embedding_matches_top10": embedding_matches_top10,
        "context_analysis": list(reverse_search_data.get("context_analysis") or []),
        "clusters": list(reverse_search_data.get("clusters") or []),
        "spread_analysis": dict(reverse_search_data.get("spread_analysis") or {"origin": "", "total_sources": 0, "risk_level": "low"}),
        "counts": counts,
        "provider_status": provider_status,
        # Backward compatibility for existing frontend consumers.
        "providers": dict(providers or {}),
    }
    return response


def _local_execution_failed(contract: dict[str, object]) -> bool:
    provider_status = dict(contract.get("provider_status") or {})
    return (
        str(provider_status.get("cloudinary") or "failed") != "ok"
        or str(provider_status.get("serpapi") or "failed") != "ok"
    )


def _compress_image_for_proxy(raw_bytes: bytes, filename: str, content_type: str) -> tuple[bytes, str, str]:
    max_bytes = 5 * 1024 * 1024
    if not raw_bytes:
        raise ValueError("Empty image payload.")

    try:
        image = Image.open(BytesIO(raw_bytes))
        image_format = str(image.format or "").lower()
        if image_format == "jpg":
            image_format = "jpeg"
        if image_format not in {"jpeg", "png", "webp"}:
            raise ValueError("Unsupported image format. Use JPG, PNG, or WEBP.")
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("Invalid image payload.") from exc

    if len(raw_bytes) <= max_bytes:
        safe_type = content_type or ("image/jpeg" if image_format == "jpeg" else f"image/{image_format}")
        return raw_bytes, filename, safe_type

    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")

    quality_steps = [85, 75, 65, 55]
    for quality in quality_steps:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", optimize=True, quality=quality)
        compressed = buffer.getvalue()
        if len(compressed) <= max_bytes:
            base = Path(filename or "upload").stem or "upload"
            return compressed, f"{base}.jpg", "image/jpeg"

    raise ValueError("Uploaded image exceeds 5MB even after compression.")


def _post_remote_reverse_search(uploaded_file, image_url: str | None) -> dict[str, object]:
    if not REMOTE_PROXY_URL:
        raise RuntimeError("REMOTE_PROXY_URL is not configured.")

    endpoint = f"{REMOTE_PROXY_URL.rstrip('/')}/proxy/reverse-search"
    last_error: Exception | None = None

    session = requests.Session()
    session.trust_env = False
    try:
        for attempt in range(1, 3):
            try:
                if uploaded_file is not None and getattr(uploaded_file, "filename", ""):
                    if hasattr(uploaded_file, "seek"):
                        uploaded_file.seek(0)
                    stream = getattr(uploaded_file, "stream", None)
                    if stream is not None and hasattr(stream, "seek"):
                        stream.seek(0)
                    payload_bytes = uploaded_file.read()
                    filename = str(getattr(uploaded_file, "filename", "upload.jpg") or "upload.jpg")
                    content_type = str(getattr(uploaded_file, "content_type", "application/octet-stream") or "application/octet-stream")
                    compressed_bytes, proxy_filename, proxy_type = _compress_image_for_proxy(payload_bytes, filename, content_type)
                    response = session.post(
                        endpoint,
                        files={"file": (proxy_filename, compressed_bytes, proxy_type)},
                        timeout=10,
                    )
                else:
                    response = session.post(
                        endpoint,
                        json={"image_url": str(image_url or "").strip()},
                        timeout=10,
                    )
                if response.status_code >= 400:
                    raise RuntimeError(f"Remote proxy HTTP {response.status_code}: {(response.text or '')[:200]}")
                payload = response.json()
                if not isinstance(payload, dict):
                    raise RuntimeError("Remote proxy returned invalid JSON payload.")
                return payload
            except Exception as exc:  # pragma: no cover - depends on network
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** (attempt - 1))
                continue
    finally:
        session.close()

    raise RuntimeError(f"Remote proxy call failed: {last_error}")


def _normalize_reverse_result(result: dict[str, object], execution: str, status: str = "success") -> dict[str, object]:
    return _contract_reverse_response(
        execution=execution,
        status=status,
        message=str(result.get("message") or "No matches found"),
        image_url=str(result.get("image_url") or result.get("public_url") or ""),
        reverse_search=dict(result.get("reverse_search") or _empty_reverse_search_payload()),
        fallback_used=bool(result.get("fallback_used", False)),
        confidence_score=float(result.get("confidence_score") or 0.0),
        providers=dict(result.get("providers") or {}),
    )


def _reverse_log(field: str, value: str) -> None:
    LOGGER.info("[ReverseSearch][%s] %s: %s", _current_request_id(), field, value)


def _valid_reverse_search_input(uploaded_file, image_url: str | None) -> bool:
    has_file = uploaded_file is not None and bool(getattr(uploaded_file, "filename", ""))
    has_url = bool(str(image_url or "").strip())
    if not has_file and not has_url:
        return False
    if not has_file:
        return True

    filename = str(getattr(uploaded_file, "filename", "") or "").strip()
    suffix = Path(filename).suffix.lower()
    if suffix not in REVERSE_SEARCH_ALLOWED_SUFFIXES:
        return False

    content_type = str(getattr(uploaded_file, "content_type", "") or "").split(";", 1)[0].strip().lower()
    if content_type and content_type not in REVERSE_SEARCH_ALLOWED_MIME_TYPES and content_type != "application/octet-stream":
        return False
    return True


@app.route("/reverse-search", methods=["POST"])
@jwt_required(AUTH_SERVICE, allow_guest=True)
def reverse_search_route():
    from ai.layer2_matching.tracking.reverse_search_service import (
        process_reverse_search,
        system_network_status,
    )

    uploaded_file = request.files.get("file")
    image_url = _request_image_url()
    print("---- REVERSE SEARCH DEBUG START ----")
    file_present = bool(uploaded_file is not None and getattr(uploaded_file, "filename", ""))
    file_name = str(getattr(uploaded_file, "filename", "") or "")
    file_type = str(getattr(uploaded_file, "content_type", "") or "")
    file_size = 0
    if uploaded_file is not None:
        stream = getattr(uploaded_file, "stream", None)
        if stream is not None and hasattr(stream, "seek") and hasattr(stream, "tell"):
            try:
                current_pos = stream.tell()
                stream.seek(0, 2)
                file_size = int(stream.tell() or 0)
                stream.seek(current_pos)
            except Exception:
                file_size = int(getattr(uploaded_file, "content_length", 0) or 0)
        else:
            file_size = int(getattr(uploaded_file, "content_length", 0) or 0)
    print("file present:", file_present)
    print("file name:", file_name or "<none>")
    print("file type:", file_type or "<unknown>")
    print("file size:", file_size)

    try:
        if uploaded_file is not None and hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        stream = getattr(uploaded_file, "stream", None)
        if stream is not None and hasattr(stream, "seek"):
            stream.seek(0)

        if not _valid_reverse_search_input(uploaded_file, image_url):
            print("after upload -> public_url:", "<none>")
            print("before SerpAPI -> image_url:", "<none>")
            print("SerpAPI response status:", "skipped")
            _reverse_log("Final", "FAILED")
            _reverse_log("Status", "FAILED")
            return jsonify({"status": "error", "message": "Invalid input"}), 400

        network_state = system_network_status(timeout_seconds=5)
        execution_mode = "LOCAL" if network_state == "open" else "REMOTE"
        _reverse_log("Mode", execution_mode)
        _reverse_log("Execution", execution_mode)

        normalized_result: dict[str, object]
        if network_state == "open":
            try:
                local_result = process_reverse_search(file=uploaded_file if file_present else None, image_url=image_url)
                normalized_result = _normalize_reverse_result(local_result, execution="local", status="success")
                provider_status = dict(normalized_result.get("provider_status") or {})
                _reverse_log("Cloudinary", "OK" if provider_status.get("cloudinary") == "ok" else "FAIL")
                _reverse_log("SerpAPI", "OK" if provider_status.get("serpapi") == "ok" else "FAIL")
                if _local_execution_failed(normalized_result):
                    _reverse_log("Mode", "REMOTE")
                    _reverse_log("Execution", "REMOTE")
                    proxy_result = _post_remote_reverse_search(uploaded_file if file_present else None, image_url=image_url)
                    normalized_result = _normalize_reverse_result(proxy_result, execution="remote", status="success")
            except Exception as local_exc:  # pragma: no cover - route safety
                LOGGER.exception("[ReverseSearch][%s] Local execution failed; attempting remote fallback.", _current_request_id())
                try:
                    _reverse_log("Mode", "REMOTE")
                    _reverse_log("Execution", "REMOTE")
                    proxy_result = _post_remote_reverse_search(uploaded_file if file_present else None, image_url=image_url)
                    normalized_result = _normalize_reverse_result(proxy_result, execution="remote", status="success")
                except Exception as proxy_exc:  # pragma: no cover - route safety
                    LOGGER.exception("[ReverseSearch][%s] Remote fallback failed after local failure.", _current_request_id())
                    normalized_result = _contract_reverse_response(
                        execution="remote_failed",
                        status="degraded",
                        message="Remote reverse-search execution failed.",
                        image_url=str(image_url or ""),
                        reverse_search=_empty_reverse_search_payload([str(local_exc), str(proxy_exc)]),
                        fallback_used=True,
                        confidence_score=0.0,
                        providers={"cloudinary": "failed", "serpapi": "failed"},
                    )
        else:
            try:
                proxy_result = _post_remote_reverse_search(uploaded_file if file_present else None, image_url=image_url)
                normalized_result = _normalize_reverse_result(proxy_result, execution="remote", status="success")
            except Exception as proxy_exc:  # pragma: no cover - route safety
                LOGGER.exception("[ReverseSearch][%s] Remote execution failed.", _current_request_id())
                normalized_result = _contract_reverse_response(
                    execution="remote_failed",
                    status="degraded",
                    message="Remote reverse-search execution failed.",
                    image_url=str(image_url or ""),
                    reverse_search=_empty_reverse_search_payload([str(proxy_exc)]),
                    fallback_used=True,
                    confidence_score=0.0,
                    providers={"cloudinary": "failed", "serpapi": "failed"},
                )

        if str(normalized_result.get("execution") or "") == "remote_failed":
            normalized_result["status"] = "degraded"
        elif _local_execution_failed(normalized_result):
            # Keep response valid but signal degraded provider status.
            normalized_result["status"] = "degraded"

        public_url = str(normalized_result.get("image_url") or "")
        provider_status = dict(normalized_result.get("provider_status") or {})
        _reverse_log("Cloudinary", "OK" if provider_status.get("cloudinary") == "ok" else "FAIL")
        _reverse_log("SerpAPI", "OK" if provider_status.get("serpapi") == "ok" else "FAIL")
        print("after upload -> public_url:", public_url or "<none>")
        print("before SerpAPI -> image_url:", public_url or "<none>")
        print("SerpAPI response status:", provider_status.get("serpapi", "failed"))
        final_status = str(normalized_result.get("status") or "degraded").upper()
        if str(normalized_result.get("execution") or "") == "remote_failed":
            final_status = "FAILED"
        _reverse_log("Final", final_status)
        _reverse_log("Status", final_status)
        exact_matches = list(normalized_result.get("exact_matches") or [])
        visual_matches = list(normalized_result.get("visual_matches") or normalized_result.get("visual_matches_top10") or [])
        embedding_matches = list(normalized_result.get("embedding_matches") or normalized_result.get("embedding_matches_top10") or [])
        reverse_payload = dict(normalized_result.get("reverse_search") or {})
        scraped_data = list(reverse_payload.get("scraped_data") or [])
        context_entries = list(normalized_result.get("context_analysis") or reverse_payload.get("context_analysis") or [])
        execution_label = str(normalized_result.get("execution") or "local").upper()
        print("Execution mode:", execution_label)
        print("Exact matches:", len(exact_matches))
        print("Visual matches:", len(visual_matches))
        print("Embedding matches:", len(embedding_matches))
        print("Scraped URLs:", len(scraped_data))
        print("Context entries:", len(context_entries))
        return jsonify(normalized_result), 200
    except Exception as exc:  # pragma: no cover - route safety
        LOGGER.exception("[ReverseSearch][%s] Route failed unexpectedly.", _current_request_id())
        print("SerpAPI response status:", "failed")
        fallback_payload = _contract_reverse_response(
            execution="remote_failed",
            status="degraded",
            message="Reverse search failed unexpectedly.",
            image_url=str(image_url or ""),
            reverse_search=_empty_reverse_search_payload([str(exc)]),
            fallback_used=True,
            confidence_score=0.0,
            providers={"cloudinary": "failed", "serpapi": "failed"},
        )
        _reverse_log("Final", "FAILED")
        _reverse_log("Status", "FAILED")
        return jsonify(fallback_payload), 200
    finally:
        print("---- REVERSE SEARCH DEBUG END ----")


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    request_id = _current_request_id()
    if isinstance(error, HTTPException):
        status_code = int(error.code or 500)
        if _is_json_api_request() or request.is_json:
            message = "Internal server error" if status_code >= 500 else str(error.description or "Request failed")
            return jsonify({"status": "error", "message": message, "request_id": request_id}), status_code
        return error

    LOGGER.exception("[GlobalError][%s] Unhandled exception on %s", request_id, request.path)
    return jsonify({"status": "error", "message": "Internal server error", "request_id": request_id}), 500


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "Uploaded file is too large. Keep it under 50 MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
