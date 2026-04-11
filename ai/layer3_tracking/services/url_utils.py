from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "source",
}
TRUSTED_DOMAIN_KEYWORDS = (
    "reuters",
    "apnews",
    "bbc",
    "nytimes",
    "wsj",
    "bloomberg",
    "npr",
    "gov",
    "edu",
)


def normalize_url(url: str) -> str:
    cleaned_url = url.strip()
    if not cleaned_url:
        return ""

    parsed = urlparse(cleaned_url)
    if not parsed.scheme or not parsed.netloc:
        return ""

    clean_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if key.lower() not in TRACKING_QUERY_KEYS and not key.lower().startswith(TRACKING_QUERY_PREFIXES)
    ]
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        query=urlencode(clean_query, doseq=True),
        fragment="",
    )
    path = normalized.path.rstrip("/") or normalized.path
    normalized = normalized._replace(path=path)
    return urlunparse(normalized)


def normalize_urls(urls: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalized_urls: list[str] = []
    for url in urls:
        cleaned = normalize_url(url)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized_urls.append(cleaned)
    return normalized_urls


def extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def is_trusted_domain(domain: str) -> bool:
    return any(keyword in domain for keyword in TRUSTED_DOMAIN_KEYWORDS)
