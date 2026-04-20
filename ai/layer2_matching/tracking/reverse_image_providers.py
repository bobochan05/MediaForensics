from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus, urljoin
from urllib.request import Request, urlopen


def _strip_html(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"<.*?>", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def _load_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return None
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#") or "=" not in cleaned:
                continue
            key, raw_value = cleaned.split("=", 1)
            if key.strip() == name:
                return raw_value.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


@dataclass
class ReverseImageCandidate:
    provider: str
    page_url: str | None = None
    media_url: str | None = None
    title: str | None = None
    snippet: str | None = None
    timestamp: str | None = None
    rank: int = 0
    frame_index: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.page_url or self.media_url or f"{self.provider}:{self.rank}:{self.frame_index}"


class ReverseImageProvider:
    name = "base"

    def is_configured(self) -> bool:
        return False

    def search_image(self, image_path: str | Path, max_results: int = 8) -> list[ReverseImageCandidate]:
        return []


class SerpApiReverseImageProvider(ReverseImageProvider):
    name = "serpapi_reverse_image"

    def __init__(self, api_key: str | None = None, timeout_seconds: int = 20) -> None:
        self.api_key = (
            api_key
            or _load_secret("SERPAPI_KEY")
            or _load_secret("SERPAPI_API_KEY")
            or _load_secret("SERP_API_KEY")
        )
        self.timeout_seconds = timeout_seconds

    def is_configured(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _as_public_url(image_path: str | Path) -> str | None:
        text = str(image_path).strip()
        if text.startswith("http://") or text.startswith("https://"):
            return text
        return None

    def _request_json(self, image_url: str) -> dict[str, object]:
        endpoint = (
            "https://serpapi.com/search.json"
            f"?engine=google_reverse_image&image_url={quote_plus(image_url)}"
            f"&api_key={quote_plus(str(self.api_key))}"
        )
        request = Request(endpoint, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def search_image(self, image_path: str | Path, max_results: int = 8) -> list[ReverseImageCandidate]:
        if not self.is_configured():
            return []

        image_url = self._as_public_url(image_path)
        if not image_url:
            return []

        try:
            payload = self._request_json(image_url=image_url)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "[SerpApiReverse] search_image failed: %s", exc,
            )
            return []

        candidates: list[ReverseImageCandidate] = []
        results = payload.get("image_results") or []
        if not isinstance(results, list):
            return []
        for index, item in enumerate(results[:max_results], start=1):
            link = item.get("link")
            if not link:
                continue
            candidate_media_url = item.get("image") or item.get("thumbnail") or None
            candidates.append(
                ReverseImageCandidate(
                    provider=self.name,
                    page_url=link,
                    media_url=candidate_media_url,
                    title=_strip_html(item.get("title")),
                    snippet=_strip_html(item.get("snippet") or item.get("displayed_link")),
                    rank=index,
                    metadata={
                        "provider_rank": index,
                        "provider_search_type": "google_reverse_image",
                        "query_image_url": image_url,
                        "provider_image_url": item.get("image"),
                        "thumbnail": item.get("thumbnail"),
                        "displayed_link": item.get("displayed_link"),
                        "cached_page_link": item.get("cached_page_link"),
                        "related_pages_link": item.get("related_pages_link"),
                    },
                )
            )

        deduped: dict[str, ReverseImageCandidate] = {}
        for candidate in candidates:
            key = candidate.key
            if key not in deduped or deduped[key].rank > candidate.rank:
                deduped[key] = candidate
        return sorted(deduped.values(), key=lambda item: item.rank)[:max_results]


class SerpApiGoogleLensProvider(ReverseImageProvider):
    name = "serpapi_google_lens"

    def __init__(self, api_key: str | None = None, timeout_seconds: int = 20) -> None:
        self.api_key = (
            api_key
            or _load_secret("SERPAPI_KEY")
            or _load_secret("SERPAPI_API_KEY")
            or _load_secret("SERP_API_KEY")
        )
        self.timeout_seconds = timeout_seconds

    def is_configured(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _as_public_url(image_path: str | Path) -> str | None:
        text = str(image_path).strip()
        if text.startswith("http://") or text.startswith("https://"):
            return text
        return None

    def _request_json(self, image_url: str, search_type: str = "visual_matches") -> dict[str, object]:
        endpoint = (
            "https://serpapi.com/search.json"
            f"?engine=google_lens&url={quote_plus(image_url)}"
            f"&type={quote_plus(search_type)}"
            f"&api_key={quote_plus(str(self.api_key))}"
        )
        request = Request(endpoint, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def search_image(self, image_path: str | Path, max_results: int = 8) -> list[ReverseImageCandidate]:
        if not self.is_configured():
            return []

        image_url = self._as_public_url(image_path)
        if not image_url:
            return []

        try:
            payload = self._request_json(image_url=image_url, search_type="visual_matches")
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "[SerpApiLens] search_image failed: %s", exc,
            )
            return []

        results = payload.get("visual_matches") or []
        if not isinstance(results, list):
            return []

        candidates: list[ReverseImageCandidate] = []
        for index, item in enumerate(results[:max_results], start=1):
            link = item.get("link")
            if not link:
                continue
            candidate_media_url = item.get("image") or item.get("thumbnail") or None
            candidates.append(
                ReverseImageCandidate(
                    provider=self.name,
                    page_url=link,
                    media_url=candidate_media_url,
                    title=_strip_html(item.get("title")),
                    snippet=_strip_html(item.get("source")),
                    rank=index,
                    metadata={
                        "provider_rank": index,
                        "provider_search_type": "google_lens_visual_matches",
                        "query_image_url": image_url,
                        "provider_image_url": item.get("image"),
                        "thumbnail": item.get("thumbnail"),
                        "source_icon": item.get("source_icon"),
                        "source_name": item.get("source"),
                        "exact_matches_hint": bool(item.get("exact_matches")),
                        "serpapi_exact_matches_link": item.get("serpapi_exact_matches_link"),
                    },
                )
            )

        deduped: dict[str, ReverseImageCandidate] = {}
        for candidate in candidates:
            key = candidate.key
            if key not in deduped or deduped[key].rank > candidate.rank:
                deduped[key] = candidate
        return sorted(deduped.values(), key=lambda item: item.rank)[:max_results]


class TinEyeReverseSearchProvider(ReverseImageProvider):
    name = "tineye"

    def __init__(
        self,
        api_url: str | None = None,
        public_key: str | None = None,
        private_key: str | None = None,
        timeout_seconds: int = 20,
    ) -> None:
        self.api_url = api_url or _load_secret("TINEYE_API_URL")
        self.public_key = public_key or _load_secret("TINEYE_API_PUBLIC_KEY")
        self.private_key = private_key or _load_secret("TINEYE_API_PRIVATE_KEY")
        self.timeout_seconds = timeout_seconds

    def is_configured(self) -> bool:
        return bool(self.api_url and self.public_key and self.private_key)

    def search_image(self, image_path: str | Path, max_results: int = 8) -> list[ReverseImageCandidate]:
        if not self.is_configured():
            return []

        image_url = str(image_path).strip()
        if not (image_url.startswith("http://") or image_url.startswith("https://")):
            return []
        auth = base64.b64encode(f"{self.public_key}:{self.private_key}".encode("utf-8")).decode("ascii")
        request = Request(
            f"{self.api_url}?image_url={quote_plus(image_url)}&limit={int(max_results)}",
            headers={
                "Authorization": f"Basic {auth}",
            },
            method="GET",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))

        results_section = body.get("result") or body.get("results") or {}
        matches = results_section.get("matches") or body.get("matches") or []

        candidates: list[ReverseImageCandidate] = []
        for index, match in enumerate(matches, start=1):
            backlinks = match.get("backlinks") or []
            page_url = None
            title = None
            if backlinks:
                first = backlinks[0]
                page_url = first.get("url") or first.get("backlink")
                title = _strip_html(first.get("title"))
            media_url = match.get("image_url") or match.get("imageUrl") or match.get("url")
            if not (page_url or media_url):
                continue
            candidates.append(
                ReverseImageCandidate(
                    provider=self.name,
                    page_url=page_url or media_url,
                    media_url=media_url,
                    title=title,
                    rank=index,
                    metadata={
                        "provider_rank": index,
                        "provider_score": match.get("score") or match.get("confidence"),
                    },
                )
            )
        return candidates[:max_results]


class BingVisualSearchProvider(ReverseImageProvider):
    name = "bing_visual_search"

    def __init__(self, endpoint: str | None = None, api_key: str | None = None, timeout_seconds: int = 20) -> None:
        self.endpoint = endpoint or _load_secret("BING_VISUAL_SEARCH_ENDPOINT") or "https://api.bing.microsoft.com/v7.0/images/visualsearch"
        self.api_key = api_key or _load_secret("BING_VISUAL_SEARCH_KEY")
        self.timeout_seconds = timeout_seconds

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def search_image(self, image_path: str | Path, max_results: int = 8) -> list[ReverseImageCandidate]:
        if not self.is_configured():
            return []

        image_url = str(image_path).strip()
        if not (image_url.startswith("http://") or image_url.startswith("https://")):
            return []
        payload = json.dumps({"imageInfo": {"url": image_url}}).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={
                "Ocp-Apim-Subscription-Key": str(self.api_key),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))

        candidates: list[ReverseImageCandidate] = []
        tags = body.get("tags") or []
        for tag in tags:
            for action in tag.get("actions") or []:
                data = action.get("data") or {}
                values = data.get("value") or []
                for index, item in enumerate(values, start=1):
                    page_url = item.get("hostPageUrl") or item.get("displayUrl") or item.get("webSearchUrl")
                    media_url = item.get("contentUrl") or item.get("thumbnailUrl") or item.get("imageId")
                    if not (page_url or media_url):
                        continue
                    candidates.append(
                        ReverseImageCandidate(
                            provider=self.name,
                            page_url=page_url or media_url,
                            media_url=media_url,
                            title=_strip_html(item.get("name")),
                            snippet=_strip_html(item.get("snippet")),
                            timestamp=item.get("datePublished"),
                            rank=index,
                            metadata={
                                "provider_rank": index,
                                "provider_action_type": action.get("actionType"),
                            },
                        )
                    )
        deduped: dict[str, ReverseImageCandidate] = {}
        for candidate in candidates:
            key = candidate.key
            if key not in deduped or deduped[key].rank > candidate.rank:
                deduped[key] = candidate
        return sorted(deduped.values(), key=lambda item: item.rank)[:max_results]


def configured_reverse_image_providers() -> list[ReverseImageProvider]:
    lens_provider = SerpApiGoogleLensProvider()
    reverse_provider = SerpApiReverseImageProvider()
    providers: list[ReverseImageProvider] = []
    # Include BOTH SerpAPI engines when both are configured (was elif — bug)
    if lens_provider.is_configured():
        providers.append(lens_provider)
    if reverse_provider.is_configured():
        providers.append(reverse_provider)
    providers.extend(
        [
            TinEyeReverseSearchProvider(),
            BingVisualSearchProvider(),
        ]
    )
    return [provider for provider in providers if provider.is_configured()]


def reverse_search_provider_status() -> dict[str, bool]:
    providers: list[ReverseImageProvider] = [
        SerpApiGoogleLensProvider(),
        SerpApiReverseImageProvider(),
        TinEyeReverseSearchProvider(),
        BingVisualSearchProvider(),
    ]
    return {provider.name: provider.is_configured() for provider in providers}


def reverse_query_cache_key(provider_name: str, image_path: str | Path) -> str:
    image_path = Path(image_path)
    digest = hashlib.sha1(image_path.read_bytes()).hexdigest()
    return f"{provider_name}_{digest}"
