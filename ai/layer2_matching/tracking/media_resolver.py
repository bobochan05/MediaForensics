from __future__ import annotations

import hashlib
import html
import json
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from ai.layer2_matching.tracking.metadata_parser import infer_platform, normalize_timestamp
from ai.shared.file_utils import ensure_dir

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


@dataclass
class PageMetadata:
    title: str | None = None
    description: str | None = None
    timestamp: str | None = None
    image_urls: list[str] = field(default_factory=list)
    video_urls: list[str] = field(default_factory=list)
    audio_urls: list[str] = field(default_factory=list)


@dataclass
class ResolvedRemoteMedia:
    page_url: str
    media_url: str | None
    local_path: Path | None
    media_type: str | None
    title: str | None = None
    caption: str | None = None
    timestamp: str | None = None
    platform: str = "external"
    metadata: dict[str, object] = field(default_factory=dict)


def _strip_html(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"<.*?>", " ", text)
    cleaned = html.unescape(re.sub(r"\s+", " ", cleaned)).strip()
    return cleaned or None


class RemoteMediaResolver:
    def __init__(self, cache_dir: str | Path, timeout_seconds: int = 20, max_download_bytes: int = 25 * 1024 * 1024) -> None:
        self.cache_dir = ensure_dir(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.max_download_bytes = int(max_download_bytes)
        self.download_dir = ensure_dir(self.cache_dir / "downloads")
        self.page_dir = ensure_dir(self.cache_dir / "pages")

    @staticmethod
    def _cache_basename(url: str) -> str:
        return hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _media_type_for(url: str | None, content_type: str | None = None) -> str | None:
        candidate = (content_type or "").lower()
        if candidate.startswith("image/"):
            return "image"
        if candidate.startswith("video/"):
            return "video"
        if candidate.startswith("audio/"):
            return "audio"

        suffix = Path(urlparse(url or "").path).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in VIDEO_EXTENSIONS:
            return "video"
        if suffix in AUDIO_EXTENSIONS:
            return "audio"
        return None

    @staticmethod
    def _extension_for(url: str, content_type: str | None = None, default: str = ".bin") -> str:
        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix:
            return suffix
        if content_type:
            guessed = mimetypes.guess_extension(content_type.split(";")[0].strip()) or default
            return guessed
        return default

    def _download_binary(self, url: str) -> tuple[Path | None, str | None]:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            content_type = response.headers.get("Content-Type")
            if content_type and "html" in content_type.lower():
                return None, content_type
            data = response.read(self.max_download_bytes + 1)
            if len(data) > self.max_download_bytes:
                return None, content_type

        extension = self._extension_for(url, content_type)
        target = self.download_dir / f"{self._cache_basename(url)}{extension}"
        if not target.exists():
            target.write_bytes(data)
        return target, content_type

    def _fetch_html(self, url: str) -> tuple[str | None, str | None]:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html,*/*;q=0.8"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            content_type = response.headers.get("Content-Type") or ""
            if "html" not in content_type.lower():
                return None, content_type
            body = response.read(self.max_download_bytes + 1)
            if len(body) > self.max_download_bytes:
                return None, content_type
            encoding = response.headers.get_content_charset() or "utf-8"
        return body.decode(encoding, errors="ignore"), content_type

    @staticmethod
    def _extract_meta_content(html_text: str, keys: list[str]) -> str | None:
        for key in keys:
            patterns = [
                re.compile(
                    rf'<meta[^>]+(?:property|name|itemprop)=["\']{re.escape(key)}["\'][^>]+content=["\'](?P<value>[^"\']+)["\']',
                    flags=re.IGNORECASE,
                ),
                re.compile(
                    rf'<meta[^>]+content=["\'](?P<value>[^"\']+)["\'][^>]+(?:property|name|itemprop)=["\']{re.escape(key)}["\']',
                    flags=re.IGNORECASE,
                ),
            ]
            for pattern in patterns:
                match = pattern.search(html_text)
                if match:
                    return html.unescape(match.group("value")).strip() or None
        return None

    @staticmethod
    def _extract_title(html_text: str) -> str | None:
        title = RemoteMediaResolver._extract_meta_content(html_text, ["og:title", "twitter:title"])
        if title:
            return title
        match = re.search(r"<title>(?P<value>.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return _strip_html(match.group("value"))
        return None

    @staticmethod
    def _extract_time_value(html_text: str) -> str | None:
        candidates = [
            RemoteMediaResolver._extract_meta_content(html_text, ["article:published_time", "og:published_time", "datePublished", "pubdate"]),
        ]
        time_match = re.search(r"<time[^>]+datetime=[\"'](?P<value>[^\"']+)[\"']", html_text, flags=re.IGNORECASE)
        if time_match:
            candidates.append(time_match.group("value"))
        ld_match = re.search(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(?P<value>.*?)</script>', html_text, flags=re.IGNORECASE | re.DOTALL)
        if ld_match:
            try:
                payload = json.loads(ld_match.group("value"))
                if isinstance(payload, dict):
                    candidates.append(payload.get("datePublished"))
                elif isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, dict) and item.get("datePublished"):
                            candidates.append(item.get("datePublished"))
                            break
            except Exception:
                pass
        for candidate in candidates:
            normalized = normalize_timestamp(candidate)
            if normalized:
                return normalized
        return None

    @staticmethod
    def _collect_urls(html_text: str, base_url: str, keys: list[str]) -> list[str]:
        urls: list[str] = []
        for key in keys:
            value = RemoteMediaResolver._extract_meta_content(html_text, [key])
            if value:
                urls.append(urljoin(base_url, value))
        return urls

    def _extract_page_metadata(self, page_url: str, html_text: str) -> PageMetadata:
        image_urls = self._collect_urls(html_text, page_url, ["og:image", "twitter:image", "og:image:url"])
        video_urls = self._collect_urls(html_text, page_url, ["og:video", "og:video:url", "twitter:player:stream"])
        audio_urls = self._collect_urls(html_text, page_url, ["og:audio", "og:audio:url"])

        if not video_urls:
            for match in re.finditer(r"<source[^>]+src=[\"'](?P<url>[^\"']+)[\"']", html_text, flags=re.IGNORECASE):
                video_urls.append(urljoin(page_url, match.group("url")))
        if not image_urls:
            for match in re.finditer(r"<img[^>]+src=[\"'](?P<url>[^\"']+)[\"']", html_text, flags=re.IGNORECASE):
                image_urls.append(urljoin(page_url, match.group("url")))
                if len(image_urls) >= 3:
                    break

        return PageMetadata(
            title=self._extract_title(html_text),
            description=self._extract_meta_content(html_text, ["description", "og:description", "twitter:description"]),
            timestamp=self._extract_time_value(html_text),
            image_urls=image_urls,
            video_urls=video_urls,
            audio_urls=audio_urls,
        )

    def resolve(
        self,
        page_url: str | None,
        media_url: str | None = None,
        title: str | None = None,
        caption: str | None = None,
        timestamp: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ResolvedRemoteMedia | None:
        metadata = dict(metadata or {})
        base_url = page_url or media_url
        if not base_url:
            return None

        downloaded_path: Path | None = None
        detected_media_type: str | None = None
        resolved_media_url = media_url
        page_metadata = PageMetadata()

        if media_url:
            try:
                downloaded_path, content_type = self._download_binary(media_url)
                detected_media_type = self._media_type_for(media_url, content_type)
            except Exception:
                downloaded_path = None
                detected_media_type = None

        html_text: str | None = None
        html_source_url = page_url or media_url
        if html_source_url:
            try:
                html_text, _ = self._fetch_html(html_source_url)
            except Exception:
                html_text = None

        if html_text:
            page_metadata = self._extract_page_metadata(html_source_url, html_text)
            if not resolved_media_url:
                if page_metadata.video_urls:
                    resolved_media_url = page_metadata.video_urls[0]
                elif page_metadata.audio_urls:
                    resolved_media_url = page_metadata.audio_urls[0]
                elif page_metadata.image_urls:
                    resolved_media_url = page_metadata.image_urls[0]

            if resolved_media_url and downloaded_path is None:
                try:
                    downloaded_path, content_type = self._download_binary(resolved_media_url)
                    detected_media_type = self._media_type_for(resolved_media_url, content_type)
                except Exception:
                    downloaded_path = None
                    detected_media_type = None

        platform = infer_platform(page_url or resolved_media_url)
        final_timestamp = normalize_timestamp(timestamp) or page_metadata.timestamp
        return ResolvedRemoteMedia(
            page_url=page_url or resolved_media_url or base_url,
            media_url=resolved_media_url,
            local_path=downloaded_path,
            media_type=detected_media_type or self._media_type_for(resolved_media_url),
            title=title or page_metadata.title,
            caption=caption or page_metadata.description,
            timestamp=final_timestamp,
            platform=platform,
            metadata={
                **metadata,
                "resolved_media_url": resolved_media_url,
                "resolved_image_urls": page_metadata.image_urls[:3],
                "resolved_video_urls": page_metadata.video_urls[:3],
                "resolved_audio_urls": page_metadata.audio_urls[:3],
            },
        )
