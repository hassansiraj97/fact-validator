"""Web access: page download, readable-text extraction and (cached) SerpAPI search."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app import config

USER_AGENT = "FactValidatorBot/1.0 (master thesis research prototype)"


async def fetch_html(url: str) -> Tuple[str, str, Optional[str]]:
    """Returns (final_url, html, error). Errors are reported instead of silently swallowed."""
    try:
        headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
        timeout = httpx.Timeout(connect=20.0, read=30.0, write=20.0, pool=30.0)
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=timeout, trust_env=False) as client:
            r = await client.get(url)
            if r.status_code >= 400:
                return str(r.url), "", f"HTTP {r.status_code}"
            return str(r.url), r.text, None
    except Exception as e:  # network errors, invalid URL, ...
        return "", "", f"{type(e).__name__}: {e}"


def extract_readable_text_from_html(html: str, url: str = "") -> str:
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html, url=url or None, include_comments=False, include_tables=False, favor_recall=True
        )
        return (extracted or "").strip()
    except Exception:
        return ""


def _cache_path(query: str, num: int):
    key = hashlib.sha1(f"{query}|{num}".encode("utf-8")).hexdigest()
    return config.SEARCH_CACHE_DIR / f"{key}.json"


async def serpapi_search(query: str, num: int = 5, use_cache: bool = True) -> List[Dict[str, Any]]:
    """Google results via SerpAPI. Results are cached on disk so experiments are reproducible
    and repeated runs do not consume API credits."""
    path = _cache_path(query, num)
    if use_cache and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))["results"]
        except Exception:
            pass

    if not config.SERPAPI_API_KEY:
        return []

    params = {
        "engine": "google", "q": query, "api_key": config.SERPAPI_API_KEY,
        "num": str(num), "hl": "en", "gl": "us",
    }
    try:
        timeout = httpx.Timeout(connect=20.0, read=45.0, write=20.0, pool=45.0)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            r = await client.get("https://serpapi.com/search", params=params)
            r.raise_for_status()
            data = r.json()
    except Exception:
        return []

    out = [
        {"title": it.get("title"), "link": it.get("link"), "snippet": it.get("snippet") or "", "date": it.get("date")}
        for it in (data.get("organic_results") or [])[:num]
    ]
    try:
        config.SEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"query": query, "num": num, "results": out}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return out
