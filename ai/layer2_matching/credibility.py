from __future__ import annotations

from urllib.parse import urlparse


HIGH_CREDIBILITY_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "nytimes.com",
    "wsj.com",
    "theguardian.com",
    "cnn.com",
    "npr.org",
}

MEDIUM_CREDIBILITY_DOMAINS = {
    "medium.com",
    "substack.com",
    "wordpress.com",
    "blogspot.com",
    "fandom.com",
}

LOW_CREDIBILITY_DOMAINS = {
    "unknown",
    "pinterest.com",
    "reddit.com",
    "tiktok.com",
    "instagram.com",
}


def _normalized_domain(url: str | None) -> str:
    if not url:
        return ""
    netloc = urlparse(str(url)).netloc.lower().removeprefix("www.")
    return netloc


def source_credibility_score(url: str | None, platform: str | None = None) -> float:
    domain = _normalized_domain(url)
    platform_name = str(platform or "").lower()

    if domain.endswith(".gov"):
        return 0.98
    if domain.endswith(".edu"):
        return 0.92
    if any(domain == item or domain.endswith(f".{item}") for item in HIGH_CREDIBILITY_DOMAINS):
        return 0.88
    if any(domain == item or domain.endswith(f".{item}") for item in MEDIUM_CREDIBILITY_DOMAINS):
        return 0.64
    if any(domain == item or domain.endswith(f".{item}") for item in LOW_CREDIBILITY_DOMAINS):
        return 0.35

    if platform_name in {"government"}:
        return 0.95
    if platform_name in {"news"}:
        return 0.82
    if platform_name in {"youtube", "video"}:
        return 0.58
    if platform_name in {"reddit", "twitter", "x", "instagram", "tiktok", "social"}:
        return 0.4
    if platform_name in {"blog"}:
        return 0.5
    if platform_name in {"local_dataset"}:
        return 0.7
    return 0.5

