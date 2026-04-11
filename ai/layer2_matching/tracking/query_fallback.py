from __future__ import annotations

import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from ai.layer2_matching.tracking.metadata_parser import OccurrenceRecord, credibility_score_for_source, infer_platform

STOPWORDS = {
    "the",
    "and",
    "with",
    "from",
    "that",
    "this",
    "have",
    "will",
    "your",
    "into",
    "about",
    "after",
    "before",
    "during",
    "image",
    "video",
    "clip",
    "media",
    "photo",
    "file",
}

LOW_SIGNAL_DOMAINS = {
    "aliexpress.com",
    "amazon.com",
    "ebay.com",
    "etsy.com",
    "flipkart.com",
    "pinterest.com",
    "temu.com",
    "walmart.com",
    "alibaba.com",
}


class QueryFallbackSearchClient:
    """Secondary internet discovery path used only when reverse-search yields nothing."""

    def __init__(self, timeout_seconds: int = 10) -> None:
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _query_from_inputs(query_hint: str | None, local_matches: list[dict[str, object]]) -> str | None:
        if query_hint and query_hint.strip():
            return query_hint.strip()

        for match in local_matches:
            candidate = str(match.get("title") or match.get("caption") or "").strip()
            if candidate:
                return candidate
        return None

    @staticmethod
    def _tokenize_text(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]{3,}", text.lower())
            if token not in STOPWORDS and not token.isdigit()
        }

    @staticmethod
    def _clean_result_url(url: str) -> str:
        decoded = html.unescape(url).strip()
        parsed = urlparse(decoded)
        if "duckduckgo.com" in parsed.netloc.lower():
            target = parse_qs(parsed.query).get("uddg")
            if target:
                return unquote(target[0])
        return decoded

    @staticmethod
    def _is_low_signal_domain(url: str | None) -> bool:
        if not url:
            return False
        netloc = urlparse(url).netloc.lower().removeprefix("www.")
        return any(netloc == domain or netloc.endswith(f".{domain}") for domain in LOW_SIGNAL_DOMAINS)

    def _score_match(
        self,
        query: str,
        title: str | None,
        snippet: str | None,
        url: str | None,
        rank: int,
        provider: str,
    ) -> tuple[float, str]:
        query_tokens = self._tokenize_text(query)
        combined_text = " ".join(part for part in [title or "", snippet or "", url or ""] if part).strip()
        text_tokens = self._tokenize_text(combined_text)
        title_tokens = self._tokenize_text(title or "")

        overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
        title_overlap = len(query_tokens & title_tokens) / max(len(query_tokens), 1)
        compact_query = re.sub(r"\s+", " ", query.lower()).strip()
        compact_text = re.sub(r"\s+", " ", combined_text.lower()).strip()
        phrase_bonus = 0.08 if compact_query and compact_query in compact_text else 0.0
        rank_bonus = max(0.0, 0.12 - 0.02 * (rank - 1))
        provider_bonus = {"google_news": 0.06, "duckduckgo": 0.05, "reddit": 0.04}.get(provider, 0.04)

        score = 0.14 + (0.28 * overlap) + (0.18 * title_overlap) + phrase_bonus + rank_bonus + provider_bonus
        score = round(max(0.05, min(score, 0.96)), 4)

        reasons: list[str] = []
        if phrase_bonus > 0:
            reasons.append("the fallback query appears directly in the result text")
        elif title_overlap >= 0.34:
            reasons.append("several fallback terms appear in the title")
        elif overlap >= 0.25:
            reasons.append("multiple fallback terms overlap with the result summary")
        else:
            reasons.append("the result matched a smaller subset of the fallback terms")
        if rank <= 2:
            reasons.append("it appeared near the top of the fallback web results")
        normalized_reasons = [reason[:1].upper() + reason[1:] + "." for reason in reasons]
        return score, " ".join(normalized_reasons)

    def _duckduckgo_search(self, query: str, limit: int = 5) -> list[OccurrenceRecord]:
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            html_text = response.read().decode("utf-8", errors="ignore")

        pattern = re.compile(
            r'class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
            r'(?:class="result__snippet"[^>]*>(?P<snippet>.*?)</(?:a|div>))',
            flags=re.IGNORECASE | re.DOTALL,
        )

        matches: list[OccurrenceRecord] = []
        for index, result in enumerate(pattern.finditer(html_text), start=1):
            result_url = self._clean_result_url(result.group("url"))
            if self._is_low_signal_domain(result_url):
                continue
            title = re.sub(r"<.*?>", "", result.group("title") or "")
            snippet = re.sub(r"<.*?>", "", result.group("snippet") or "")
            platform = infer_platform(result_url)
            score, reason = self._score_match(query, title, snippet, result_url, rank=index, provider="duckduckgo")
            matches.append(
                OccurrenceRecord(
                    entry_id=f"ddg-{index}-{hashlib.sha1(result_url.encode('utf-8')).hexdigest()[:12]}",
                    source_type="external",
                    platform=platform,
                    url=result_url,
                    title=html.unescape(title).strip() or None,
                    caption=html.unescape(snippet).strip() or None,
                    credibility_score=credibility_score_for_source(platform, result_url),
                    fused_similarity=score,
                    metadata={"provider": "duckduckgo_fallback", "query": query, "match_reason": reason},
                )
            )
            if len(matches) >= limit:
                break
        return matches

    def _google_news_search(self, query: str, limit: int = 4) -> list[OccurrenceRecord]:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = response.read()

        root = ET.fromstring(payload)
        results: list[OccurrenceRecord] = []
        for index, item in enumerate(root.findall("./channel/item"), start=1):
            result_url = self._clean_result_url(item.findtext("link", "").strip())
            if not result_url:
                continue
            if self._is_low_signal_domain(result_url):
                continue

            title = (item.findtext("title") or "").strip()
            description = re.sub(r"<.*?>", " ", item.findtext("description") or "")
            timestamp_text = (item.findtext("pubDate") or "").strip()
            timestamp: str | None = None
            if timestamp_text:
                try:
                    timestamp = parsedate_to_datetime(timestamp_text).astimezone(timezone.utc).isoformat()
                except (TypeError, ValueError, OverflowError):
                    timestamp = None

            platform = infer_platform(result_url, fallback="news")
            score, reason = self._score_match(query, title, description, result_url, rank=index, provider="google_news")
            results.append(
                OccurrenceRecord(
                    entry_id=f"gnews-{index}-{hashlib.sha1(result_url.encode('utf-8')).hexdigest()[:12]}",
                    source_type="external",
                    platform=platform,
                    url=result_url,
                    timestamp=timestamp,
                    title=title or None,
                    caption=html.unescape(description).strip() or None,
                    credibility_score=credibility_score_for_source(platform, result_url),
                    fused_similarity=score,
                    metadata={"provider": "google_news_fallback", "query": query, "match_reason": reason},
                )
            )
            if len(results) >= limit:
                break
        return results

    def _reddit_search(self, query: str, limit: int = 3) -> list[OccurrenceRecord]:
        url = f"https://www.reddit.com/search.json?q={quote_plus(query)}&sort=new&limit={limit}"
        request = Request(url, headers={"User-Agent": "deepfake-detector-layer2/1.0"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))

        results: list[OccurrenceRecord] = []
        for index, child in enumerate(payload.get("data", {}).get("children", []), start=1):
            data = child.get("data", {})
            permalink = data.get("permalink")
            if not permalink:
                continue
            result_url = f"https://www.reddit.com{permalink}"
            title = str(data.get("title") or "").strip()
            caption = str(data.get("selftext") or "").strip()[:400] or None
            score, reason = self._score_match(query, title, caption, result_url, rank=index, provider="reddit")
            results.append(
                OccurrenceRecord(
                    entry_id=f"reddit-{data.get('id', '')}",
                    source_type="external",
                    platform="reddit",
                    url=result_url,
                    timestamp=datetime.fromtimestamp(float(data.get("created_utc", 0.0)), tz=timezone.utc).isoformat(),
                    title=title or None,
                    caption=caption,
                    credibility_score=credibility_score_for_source("reddit", result_url),
                    fused_similarity=score,
                    metadata={"provider": "reddit_fallback", "subreddit": data.get("subreddit"), "query": query, "match_reason": reason},
                )
            )
        return results

    @staticmethod
    def _mock_results(query: str | None, limit: int = 4) -> list[OccurrenceRecord]:
        seed = int(hashlib.sha1(f"{query or ''}".encode("utf-8")).hexdigest()[:8], 16)
        base_time = datetime.now(timezone.utc) - timedelta(days=(seed % 30) + 1)
        title_hint = query or "candidate content"
        templates = [
            ("twitter", f"{title_hint} clip spreads on X", "Short post resharing the media with minimal context."),
            ("reddit", f"Users debate whether {title_hint} is authentic", "Discussion thread comparing versions of the media."),
            ("news", f"Explainer about viral media: {title_hint}", "Article analyzing whether the circulating content is misleading."),
            ("blog", f"Blog roundup referencing {title_hint}", "Low-credibility repost that amplifies the content."),
        ]

        results: list[OccurrenceRecord] = []
        for index, (platform, title, caption) in enumerate(templates[:limit], start=1):
            timestamp = (base_time + timedelta(hours=index * (seed % 5 + 2))).isoformat()
            if platform == "twitter":
                url = f"https://x.com/mock/status/{seed + index}"
            elif platform == "reddit":
                url = f"https://www.reddit.com/r/mock/comments/{seed + index}/layer2_demo/"
            elif platform == "news":
                url = f"https://news.example.com/story/{seed + index}"
            else:
                url = f"https://blog.example.com/post/{seed + index}"
            results.append(
                OccurrenceRecord(
                    entry_id=f"mock-{platform}-{seed + index}",
                    source_type="external",
                    platform=platform,
                    url=url,
                    timestamp=timestamp,
                    title=title,
                    caption=caption,
                    credibility_score=credibility_score_for_source(platform, url),
                    fused_similarity=round(max(0.25, 0.86 - (index * 0.09)), 4),
                    is_mock=True,
                    metadata={
                        "provider": "mock_fallback",
                        "query": query,
                        "match_reason": "Fallback demo result generated because no live reverse or text results were reachable.",
                    },
                )
            )
        return results

    def search(
        self,
        query_hint: str | None,
        local_matches: list[dict[str, object]],
        max_results: int = 6,
        allow_mock_fallback: bool = False,
    ) -> list[OccurrenceRecord]:
        query = self._query_from_inputs(query_hint, local_matches)
        if not query:
            return self._mock_results(None, limit=min(max_results, 4)) if allow_mock_fallback else []

        results: list[OccurrenceRecord] = []
        try:
            results.extend(self._google_news_search(query, limit=max_results))
        except Exception:
            pass
        try:
            results.extend(self._duckduckgo_search(query, limit=max_results))
        except Exception:
            pass
        try:
            results.extend(self._reddit_search(query, limit=max_results))
        except Exception:
            pass

        deduped: dict[str, OccurrenceRecord] = {}
        for result in results:
            key = result.url or result.entry_id
            if key not in deduped or deduped[key].fused_similarity < result.fused_similarity:
                deduped[key] = result

        if deduped:
            ranked = sorted(
                (item for item in deduped.values() if item.fused_similarity >= 0.18),
                key=lambda item: (item.fused_similarity, item.credibility_score),
                reverse=True,
            )
            return ranked[:max_results]
        return self._mock_results(query, limit=min(max_results, 4)) if allow_mock_fallback else []
