from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cloudinary
import cloudinary.uploader
import requests
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = APP_DIR / ".env"
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
DEFAULT_TIMEOUT_SECONDS = 25
TRUSTED_DOMAINS = {
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "cnn.com",
    "nytimes.com",
    "reuters.com",
    "theguardian.com",
    "washingtonpost.com",
    "youtube.com",
    "youtu.be",
    "wikipedia.org",
}
_ENV_LOADED = False


class ReverseSearchError(RuntimeError):
    """Base exception for reverse-search failures."""


class ReverseSearchInputError(ReverseSearchError):
    """Raised when the user input is missing or invalid."""


def _load_environment() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    load_dotenv(ENV_PATH)
    _ENV_LOADED = True


def _get_env(*names: str) -> str | None:
    _load_environment()
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return None


def _log_json(title: str, payload: Any) -> None:
    try:
        formatted = json.dumps(payload, indent=2, ensure_ascii=True, default=str)
    except TypeError:
        formatted = str(payload)
    LOGGER.info("%s\n%s", title, formatted)
    if not LOGGER.handlers:
        print(f"{title}\n{formatted}")


def _is_http_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_url(value: str | None) -> str:
    candidate = (value or "").strip()
    if not _is_http_url(candidate):
        raise ReverseSearchInputError("Provide a valid public HTTP or HTTPS image URL.")
    return candidate


def _extract_domain(value: str | None) -> str | None:
    if not value or not _is_http_url(value):
        return None
    parsed = urlparse(value)
    return parsed.netloc.lower().removeprefix("www.")


def _first_text(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _cloudinary_ready() -> bool:
    return all(
        [
            _get_env("CLOUDINARY_CLOUD_NAME"),
            _get_env("CLOUDINARY_API_KEY"),
            _get_env("CLOUDINARY_API_SECRET"),
        ]
    )


def _configure_cloudinary() -> None:
    if not _cloudinary_ready():
        raise ReverseSearchInputError("Cloudinary credentials are not configured in the environment.")
    cloudinary.config(
        cloud_name=_get_env("CLOUDINARY_CLOUD_NAME"),
        api_key=_get_env("CLOUDINARY_API_KEY"),
        api_secret=_get_env("CLOUDINARY_API_SECRET"),
        secure=True,
    )


def upload_to_cloudinary(file) -> str:
    """Upload a Flask file object to Cloudinary and return a secure public URL."""

    if file is None or not getattr(file, "filename", ""):
        raise ReverseSearchInputError("Upload an image file or provide a public image URL.")

    _configure_cloudinary()

    upload_target = getattr(file, "stream", file)
    try:
        if hasattr(upload_target, "seek"):
            upload_target.seek(0)
        result = cloudinary.uploader.upload(
            upload_target,
            folder="deepfake-detector/reverse-search",
            resource_type="image",
            use_filename=True,
            unique_filename=True,
            overwrite=False,
            filename_override=Path(str(file.filename)).name,
            tags=["deepfake-detector", "reverse-search", "layer2"],
        )
        secure_url = result.get("secure_url")
        if not secure_url:
            raise ReverseSearchError("Cloudinary upload did not return a secure URL.")
        return _validate_url(secure_url)
    except ReverseSearchError:
        raise
    except Exception as exc:  # pragma: no cover - depends on external service
        LOGGER.exception("Cloudinary upload failed for %s", getattr(file, "filename", "<unknown>"))
        raise ReverseSearchError("Cloudinary upload failed. Check the credentials and file contents.") from exc


def upload_path_to_cloudinary(file_path: str | Path, filename_override: str | None = None) -> str:
    """Upload a local image path to Cloudinary and return a secure public URL."""

    path_obj = Path(file_path)
    if not path_obj.exists() or not path_obj.is_file():
        raise ReverseSearchInputError("Provide a valid local image file path.")

    _configure_cloudinary()

    try:
        result = cloudinary.uploader.upload(
            str(path_obj),
            folder="deepfake-detector/reverse-search",
            resource_type="image",
            use_filename=True,
            unique_filename=True,
            overwrite=False,
            filename_override=filename_override or path_obj.name,
            tags=["deepfake-detector", "reverse-search", "layer2"],
        )
        secure_url = result.get("secure_url")
        if not secure_url:
            raise ReverseSearchError("Cloudinary upload did not return a secure URL.")
        return _validate_url(secure_url)
    except ReverseSearchError:
        raise
    except Exception as exc:  # pragma: no cover - depends on external service
        LOGGER.exception("Cloudinary upload failed for path %s", path_obj)
        raise ReverseSearchError("Cloudinary upload failed. Check the credentials and file contents.") from exc


def ensure_public_url(file=None, image_url: str | None = None) -> str:
    """Return a public URL for either an uploaded file or a provided URL."""

    if file is not None and getattr(file, "filename", ""):
        return upload_to_cloudinary(file)
    if image_url:
        return _validate_url(image_url)
    raise ReverseSearchInputError("Provide either an uploaded image file or a public image URL.")


def _serpapi_request(engine: str, image_url: str, *, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    api_key = _get_env("SERPAPI_KEY", "SERPAPI_API_KEY", "SERP_API_KEY")
    if not api_key:
        raise ReverseSearchError("SERPAPI_KEY is not configured in the environment.")

    params: dict[str, Any] = {"engine": engine, "api_key": api_key}
    if engine == "google_reverse_image":
        params["image_url"] = _validate_url(image_url)
    elif engine == "google_lens":
        params["url"] = _validate_url(image_url)
    else:
        raise ReverseSearchError(f"Unsupported SerpApi engine: {engine}")

    try:
        response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        LOGGER.exception("SerpApi %s request timed out.", engine)
        raise ReverseSearchError(f"{engine} request to SerpApi timed out.") from exc
    except requests.RequestException as exc:
        LOGGER.exception("SerpApi %s request failed.", engine)
        raise ReverseSearchError(f"{engine} request to SerpApi failed.") from exc
    except ValueError as exc:
        LOGGER.exception("SerpApi %s returned invalid JSON.", engine)
        raise ReverseSearchError(f"{engine} returned invalid JSON.") from exc

    _log_json(f"SerpApi response ({engine})", payload)

    if payload.get("error"):
        LOGGER.error("SerpApi %s returned an error: %s", engine, payload["error"])
        raise ReverseSearchError(str(payload["error"]))

    search_status = str((payload.get("search_metadata") or {}).get("status") or "").lower()
    if search_status == "error":
        message = payload.get("error") or "SerpApi search_metadata.status=Error"
        LOGGER.error("SerpApi %s metadata error: %s", engine, message)
        raise ReverseSearchError(str(message))

    return payload


def reverse_image_search(image_url: str) -> dict[str, Any]:
    """Run the primary SerpApi reverse-image request."""

    return _serpapi_request(engine="google_reverse_image", image_url=image_url)


def _normalize_result_item(item: Any, *, bucket: str, rank: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    page_url = _first_text(
        item.get("link"),
        item.get("page_url"),
        item.get("page"),
        item.get("source"),
        item.get("website"),
        item.get("redirect_link"),
        item.get("hostPageUrl"),
    )
    image_url = _first_text(
        item.get("image"),
        item.get("image_url"),
        item.get("thumbnail"),
        item.get("thumbnail_url"),
        item.get("thumbnailUrl"),
        item.get("original"),
        item.get("original_image"),
        item.get("contentUrl"),
    )
    if page_url and not _is_http_url(page_url):
        page_url = None
    if image_url and not _is_http_url(image_url):
        image_url = None

    title = _first_text(item.get("title"), item.get("name"), item.get("source_name"))
    snippet = _first_text(
        item.get("snippet"),
        item.get("description"),
        item.get("displayed_link"),
        item.get("source"),
    )
    domain = _extract_domain(page_url) or _extract_domain(image_url)
    trusted = domain in TRUSTED_DOMAINS if domain else False

    normalized = {
        "bucket": bucket,
        "rank": rank,
        "title": title,
        "snippet": snippet,
        "page_url": page_url,
        "image_url": image_url,
        "domain": domain,
        "source": domain or _first_text(item.get("displayed_link"), item.get("source")),
        "trusted_domain": trusted,
        "raw": item,
    }
    if not any([normalized["page_url"], normalized["image_url"], normalized["title"], normalized["snippet"]]):
        return None
    return normalized


def _build_sources(matches: list[dict[str, Any]], knowledge_graph: dict[str, Any] | None) -> list[dict[str, Any]]:
    domains = [match["domain"] for match in matches if match.get("domain")]
    counts = Counter(domains)
    sources = [
        {
            "domain": domain,
            "count": count,
            "trusted_domain": domain in TRUSTED_DOMAINS,
        }
        for domain, count in counts.most_common()
    ]
    if knowledge_graph:
        kg_source = _extract_domain(_first_text(knowledge_graph.get("source"), knowledge_graph.get("website")))
        if kg_source and all(source["domain"] != kg_source for source in sources):
            sources.append(
                {
                    "domain": kg_source,
                    "count": 1,
                    "trusted_domain": kg_source in TRUSTED_DOMAINS,
                }
            )
    return sources


def parse_serpapi_results(results: dict[str, Any]) -> dict[str, Any]:
    """Normalize the useful SerpApi response sections into a stable structure."""

    image_results_raw = results.get("image_results") or []
    inline_images_raw = results.get("inline_images") or []
    visual_matches_raw = (
        results.get("visual_matches")
        or results.get("exact_matches")
        or results.get("similar_images")
        or []
    )
    related_content_raw = results.get("related_content") or []
    knowledge_graph_raw = results.get("knowledge_graph") or {}

    image_results = [
        item
        for index, raw in enumerate(image_results_raw, start=1)
        if (item := _normalize_result_item(raw, bucket="image_results", rank=index)) is not None
    ]
    inline_images = [
        item
        for index, raw in enumerate(inline_images_raw, start=1)
        if (item := _normalize_result_item(raw, bucket="inline_images", rank=index)) is not None
    ]
    visual_matches = [
        item
        for index, raw in enumerate(visual_matches_raw, start=1)
        if (item := _normalize_result_item(raw, bucket="visual_matches", rank=index)) is not None
    ]
    related_content = [
        item
        for index, raw in enumerate(related_content_raw, start=1)
        if (item := _normalize_result_item(raw, bucket="related_content", rank=index)) is not None
    ]

    top_matches = (visual_matches + image_results + related_content)[:10]
    similar_images = inline_images[:12]

    knowledge_graph = None
    if isinstance(knowledge_graph_raw, dict) and knowledge_graph_raw:
        knowledge_graph = {
            "title": _first_text(knowledge_graph_raw.get("title"), knowledge_graph_raw.get("name")),
            "subtitle": _first_text(knowledge_graph_raw.get("subtitle"), knowledge_graph_raw.get("type")),
            "description": _first_text(knowledge_graph_raw.get("description")),
            "source": _first_text(knowledge_graph_raw.get("source"), knowledge_graph_raw.get("website")),
        }

    all_matches = top_matches + similar_images
    sources = _build_sources(all_matches, knowledge_graph)

    return {
        "matches_found": bool(top_matches or similar_images),
        "top_matches": top_matches,
        "similar_images": similar_images,
        "sources": sources,
        "image_results": image_results,
        "inline_images": inline_images,
        "visual_matches": visual_matches,
        "knowledge_graph": knowledge_graph,
    }


def fallback_search(image_url: str) -> dict[str, Any]:
    """Fallback to Google Lens and include placeholders for future local verification methods."""

    try:
        raw_results = _serpapi_request(engine="google_lens", image_url=image_url)
        parsed = parse_serpapi_results(raw_results)
        parsed["engine"] = "google_lens"
        parsed["placeholder_methods"] = [
            {
                "name": "perceptual_hash",
                "available": False,
                "reason": "Remote candidate pHash verification is reserved for the next verification pass.",
            },
            {
                "name": "embedding_similarity",
                "available": False,
                "reason": "Remote embedding verification is handled in the Layer 2 verification pipeline, not in this route yet.",
            },
        ]
        return parsed
    except ReverseSearchError as exc:
        LOGGER.exception("Fallback Google Lens search failed.")
        return {
            "matches_found": False,
            "top_matches": [],
            "similar_images": [],
            "sources": [],
            "image_results": [],
            "inline_images": [],
            "visual_matches": [],
            "knowledge_graph": None,
            "engine": "google_lens",
            "errors": [str(exc)],
            "placeholder_methods": [
                {
                    "name": "perceptual_hash",
                    "available": False,
                    "reason": "Google Lens fallback failed before local verification placeholders could run.",
                },
                {
                    "name": "embedding_similarity",
                    "available": False,
                    "reason": "Google Lens fallback failed before remote verification placeholders could run.",
                },
            ],
        }


def compute_confidence(results: dict[str, Any]) -> float:
    """Heuristic confidence score for how strong the reverse-search evidence looks."""

    top_matches = results.get("top_matches") or []
    similar_images = results.get("similar_images") or []
    sources = results.get("sources") or []
    total_matches = len(top_matches) + len(similar_images)

    if total_matches == 0:
        return 0.0

    volume_score = min(total_matches / 10.0, 1.0)
    trusted_sources = sum(1 for source in sources if source.get("trusted_domain"))
    trusted_score = min(trusted_sources / max(len(sources), 1), 1.0)
    populated_sections = sum(
        bool(results.get(section))
        for section in ("top_matches", "similar_images", "sources", "knowledge_graph")
    )
    consistency_score = min(populated_sections / 4.0, 1.0)

    score = (0.45 * volume_score) + (0.25 * trusted_score) + (0.30 * consistency_score)
    return round(min(max(score, 0.0), 1.0), 3)


def _dedupe_matches(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in items:
        key = item.get("page_url") or item.get("image_url") or json.dumps(item, sort_keys=True)
        if key not in deduped:
            deduped[key] = item
            continue
        existing = deduped[key]
        if (item.get("rank") or 10_000) < (existing.get("rank") or 10_000):
            deduped[key] = item
    return list(deduped.values())


def _merge_parsed_results(primary: dict[str, Any], fallback: dict[str, Any] | None) -> dict[str, Any]:
    fallback = fallback or {}
    top_matches = _dedupe_matches((primary.get("top_matches") or []) + (fallback.get("top_matches") or []))
    similar_images = _dedupe_matches((primary.get("similar_images") or []) + (fallback.get("similar_images") or []))
    sources = _build_sources(top_matches + similar_images, primary.get("knowledge_graph") or fallback.get("knowledge_graph"))
    errors = []
    errors.extend(primary.get("errors") or [])
    errors.extend(fallback.get("errors") or [])
    return {
        "matches_found": bool(top_matches or similar_images),
        "top_matches": sorted(top_matches, key=lambda item: item.get("rank") or 10_000)[:10],
        "similar_images": sorted(similar_images, key=lambda item: item.get("rank") or 10_000)[:12],
        "sources": sources,
        "image_results": primary.get("image_results") or [],
        "inline_images": primary.get("inline_images") or [],
        "visual_matches": _dedupe_matches((primary.get("visual_matches") or []) + (fallback.get("visual_matches") or [])),
        "knowledge_graph": primary.get("knowledge_graph") or fallback.get("knowledge_graph"),
        "primary_engine": "google_reverse_image",
        "fallback_engine": fallback.get("engine") if fallback else None,
        "errors": errors,
        "placeholder_methods": fallback.get("placeholder_methods") or [],
    }


def process_reverse_search(file=None, image_url: str | None = None) -> dict[str, Any]:
    """Execute the full reverse-search pipeline for a file upload or public URL."""

    public_url = ensure_public_url(file=file, image_url=image_url)
    primary_errors: list[str] = []

    try:
        primary_raw = reverse_image_search(public_url)
        primary_parsed = parse_serpapi_results(primary_raw)
    except ReverseSearchError as exc:
        LOGGER.exception("Primary reverse image search failed.")
        primary_errors.append(str(exc))
        primary_parsed = {
            "matches_found": False,
            "top_matches": [],
            "similar_images": [],
            "sources": [],
            "image_results": [],
            "inline_images": [],
            "visual_matches": [],
            "knowledge_graph": None,
            "errors": primary_errors,
        }
    else:
        if primary_errors:
            primary_parsed["errors"] = primary_errors

    fallback_used = not primary_parsed.get("matches_found", False)
    fallback_parsed = fallback_search(public_url) if fallback_used else None
    combined = _merge_parsed_results(primary_parsed, fallback_parsed)
    confidence_score = compute_confidence(combined)

    return {
        "image_url": public_url,
        "reverse_search": combined,
        "fallback_used": fallback_used,
        "confidence_score": confidence_score,
        "message": "Matches found" if combined.get("matches_found") else "No matches found",
    }
