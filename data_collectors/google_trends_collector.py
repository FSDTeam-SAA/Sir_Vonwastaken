"""
data_collectors/google_trends_collector.py

Module 1 (Google Trends part) — uses Google's Trending Now RSS export and
`pytrends` (an unofficial Google Trends client; no API key required) to pull:
  - currently trending search terms for a region
  - interest-over-time for configured keywords (to compute growth velocity)
  - related/rising queries for a keyword (to surface adjacent opportunities)
"""
from __future__ import annotations

import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional
from xml.etree import ElementTree

import requests
from pytrends.request import TrendReq

from config.settings import settings
from database.mongodb import upsert
from utils.logger import logger

_TRENDING_RSS_URL = "https://trends.google.com/trending/rss"
_TRENDING_RSS_NAMESPACE = {"ht": "https://trends.google.com/trending/rss"}
_GOOGLE_REQUEST_DELAY_SECONDS = 1.0
_RATE_LIMIT_RETRY_SECONDS = 15.0


def get_trends_client() -> TrendReq:
    """Return an isolated client because TrendReq payload state is mutable."""
    return TrendReq(hl="en-US", tz=360)


def _store_raw(doc: Dict) -> None:
    upsert(
        "raw_content",
        {"platform": "google_trends", "external_id": doc["external_id"]},
        {
            **doc,
            "platform": "google_trends",
            "collected_at": datetime.utcnow(),
        },
    )


def get_trending_searches(
    geo: Optional[str] = None,
    errors: Optional[List[str]] = None,
) -> List[Dict]:
    """Collect current Trending Now terms from Google's official RSS export.

    ``pytrends.trending_searches()`` still targets Google's retired
    ``/hottrends/visualize/internal/data`` URL, which now returns HTTP 404.
    The Trending Now RSS feed is the current export exposed by Google Trends.
    """
    resolved_geo = (geo or settings.google_trends_geo or "US").upper()
    try:
        response = requests.get(
            _TRENDING_RSS_URL,
            params={"geo": resolved_geo},
            headers={
                "User-Agent": "Mozilla/5.0 "
                "(compatible; ContentTrendAssistant/1.0)"
            },
            timeout=15,
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
    except (requests.RequestException, ElementTree.ParseError) as exc:
        message = (
            f"Trending Now RSS failed for geo={resolved_geo} "
            f"({_exception_label(exc)})."
        )
        logger.error("{} Full error: {}", message, exc)
        if errors is not None:
            errors.append(message)
        return []

    docs = []
    for rank, item in enumerate(root.findall("./channel/item")):
        keyword = (item.findtext("title") or "").strip()
        if not keyword:
            continue

        raw_published_at = (item.findtext("pubDate") or "").strip()
        try:
            published_at = parsedate_to_datetime(raw_published_at).isoformat()
        except (TypeError, ValueError):
            published_at = raw_published_at

        doc = {
            "external_id": (
                f"{resolved_geo}:{datetime.utcnow().date().isoformat()}:{keyword}"
            ),
            "record_kind": "trending_search",
            "keyword": keyword,
            "rank": rank,
            "geo": resolved_geo,
            "approx_traffic": (
                item.findtext(
                    "ht:approx_traffic",
                    default="",
                    namespaces=_TRENDING_RSS_NAMESPACE,
                )
                or ""
            ).strip(),
            "published_at": published_at,
            "source_url": f"https://trends.google.com/trending?geo={resolved_geo}",
        }
        docs.append(doc)
        _store_raw(doc)

    if not docs:
        message = f"Trending Now RSS returned no items for geo={resolved_geo}."
        logger.warning(message)
        if errors is not None:
            errors.append(message)

    logger.info(f"Collected {len(docs)} Trending Now searches for geo={resolved_geo}.")
    return docs


def get_interest_over_time(
    keywords: List[str],
    timeframe: str = "now 7-d",
    errors: Optional[List[str]] = None,
    client: Optional[TrendReq] = None,
) -> Dict[str, List[Dict]]:
    """
    Module 1 + growth-velocity input for the ranking engine: pulls the
    interest-over-time series for up to 5 keywords at a time (a pytrends/
    Google Trends API limit).
    """
    client = client or get_trends_client()
    results: Dict[str, List[Dict]] = {}

    batches = [keywords[index : index + 5] for index in range(0, len(keywords), 5)]
    for batch_index, batch in enumerate(batches):
        df = None
        failure = None
        for attempt in range(2):
            try:
                client.build_payload(
                    batch,
                    timeframe=timeframe,
                    geo=settings.google_trends_geo,
                )
                df = client.interest_over_time()
                break
            except Exception as exc:  # noqa: BLE001 - pytrends uses generic errors
                failure = exc
                if _response_status_code(exc) == 429 and attempt == 0:
                    logger.warning(
                        "Google Trends rate-limited interest-over-time for {}; "
                        "retrying in {} seconds.",
                        batch,
                        _RATE_LIMIT_RETRY_SECONDS,
                    )
                    time.sleep(_RATE_LIMIT_RETRY_SECONDS)
                    continue
                break

        if df is None:
            message = (
                f"Interest-over-time failed for {batch} "
                f"({_exception_label(failure)})."
            )
            logger.error("{} Full error: {}", message, failure)
            if errors is not None:
                errors.append(message)
            if failure is not None and _response_status_code(failure) == 429:
                remaining = [
                    keyword
                    for later_batch in batches[batch_index + 1 :]
                    for keyword in later_batch
                ]
                if remaining and errors is not None:
                    errors.append(
                        f"Interest-over-time skipped after rate limit for {remaining}"
                    )
                break
            continue

        if df.empty:
            if batch_index < len(batches) - 1:
                time.sleep(_GOOGLE_REQUEST_DELAY_SECONDS)
            continue

        for kw in batch:
            if kw not in df.columns:
                continue
            series = [
                {"timestamp": ts.isoformat(), "value": int(val)}
                for ts, val in df[kw].items()
            ]
            results[kw] = series
            _store_raw(
                {
                    "external_id": f"interest:{kw}:{timeframe}",
                    "record_kind": "interest_over_time",
                    "keyword": kw,
                    "timeframe": timeframe,
                    "series": series,
                }
            )

        if batch_index < len(batches) - 1:
            time.sleep(_GOOGLE_REQUEST_DELAY_SECONDS)

    logger.info(f"Collected interest-over-time for {list(results.keys())}.")
    return results


def _response_status_code(exc: Exception) -> Optional[int]:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _exception_label(exc: Optional[Exception]) -> str:
    if exc is None:
        return "unknown error"
    status_code = _response_status_code(exc)
    if status_code is not None:
        return f"HTTP {status_code}"
    return type(exc).__name__


def get_related_queries_batch(
    keywords: List[str],
    errors: Optional[List[str]] = None,
    client: Optional[TrendReq] = None,
) -> Dict[str, Dict[str, List[Dict]]]:
    """Collect related queries sequentially with pacing and one 429 retry.

    ``pytrends.related_queries()`` performs a separate Google request for each
    keyword even when the payload contains several keywords. Processing one at
    a time preserves earlier successes if a later request is rate-limited.
    """
    client = client or get_trends_client()
    results: Dict[str, Dict[str, List[Dict]]] = {}

    for keyword_index, keyword in enumerate(keywords):
        related = None
        failure = None
        for attempt in range(2):
            try:
                client.build_payload([keyword], geo=settings.google_trends_geo)
                related = client.related_queries()
                break
            except Exception as exc:  # noqa: BLE001 - pytrends uses generic errors
                failure = exc
                if _response_status_code(exc) == 429 and attempt == 0:
                    logger.warning(
                        "Google Trends rate-limited related queries for {!r}; "
                        "retrying in {} seconds.",
                        keyword,
                        _RATE_LIMIT_RETRY_SECONDS,
                    )
                    time.sleep(_RATE_LIMIT_RETRY_SECONDS)
                    continue
                break

        if related is None:
            message = (
                f"Related queries failed for {keyword!r} "
                f"({_exception_label(failure)})."
            )
            logger.error("{} Full error: {}", message, failure)
            if errors is not None:
                errors.append(message)

            # A repeated 429 means the current Google quota window is still
            # closed. Stop here instead of sleeping/retrying every remaining
            # keyword; a later collection run can fill the missing records.
            if failure is not None and _response_status_code(failure) == 429:
                remaining = keywords[keyword_index + 1 :]
                if remaining and errors is not None:
                    errors.append(
                        f"Related queries skipped after rate limit for {remaining}"
                    )
                break
        else:
            if keyword not in related:
                message = f"Related queries response omitted {keyword!r}."
                logger.warning(message)
                if errors is not None:
                    errors.append(message)
                if keyword_index < len(keywords) - 1:
                    time.sleep(_GOOGLE_REQUEST_DELAY_SECONDS)
                continue

            data = related.get(keyword) or {}
            top_df, rising_df = data.get("top"), data.get("rising")
            result = {
                "top": top_df.to_dict("records") if top_df is not None else [],
                "rising": rising_df.to_dict("records") if rising_df is not None else [],
            }
            results[keyword] = result
            _store_raw(
                {
                    "external_id": f"related:{keyword}",
                    "record_kind": "related_queries",
                    "keyword": keyword,
                    **result,
                }
            )

        if keyword_index < len(keywords) - 1:
            time.sleep(_GOOGLE_REQUEST_DELAY_SECONDS)

    return results


def get_related_queries(keyword: str) -> Dict[str, List[Dict]]:
    """Compatibility wrapper for collecting related queries for one keyword."""
    return get_related_queries_batch([keyword]).get(
        keyword,
        {"top": [], "rising": []},
    )


def run_full_collection() -> Dict[str, object]:
    """Collect trending searches and analytics for configured keywords."""
    errors: List[str] = []
    keywords = list(dict.fromkeys(settings.google_trends_keywords))
    trending = get_trending_searches(errors=errors)

    interest: Dict[str, List[Dict]] = {}
    related: Dict[str, Dict[str, List[Dict]]] = {}
    if keywords:
        try:
            client = get_trends_client()
        except Exception as exc:  # noqa: BLE001 - initialization uses the network
            message = (
                "Google Trends keyword client initialization failed "
                f"({_exception_label(exc)})."
            )
            logger.error("{} Full error: {}", message, exc)
            errors.append(message)
        else:
            interest = get_interest_over_time(keywords, errors=errors, client=client)
            related = get_related_queries_batch(keywords, errors=errors, client=client)

            missing_interest = [
                keyword for keyword in keywords if keyword not in interest
            ]
            if missing_interest:
                errors.append(
                    f"No interest-over-time data returned for {missing_interest}"
                )

            missing_related = [
                keyword for keyword in keywords if keyword not in related
            ]
            if missing_related and not any(
                "rate limit" in error.lower() for error in errors
            ):
                errors.append(f"No related-query data returned for {missing_related}")
    else:
        message = (
            "GOOGLE_TRENDS_KEYWORDS is empty; keyword and related-query "
            "collection was skipped."
        )
        logger.warning(message)
        errors.append(message)

    result = {
        "status": "ok" if not errors else "partial",
        "trending_searches": len(trending),
        "keywords_requested": len(keywords),
        "keywords_tracked": len(interest),
        "related_queries_tracked": len(related),
        "errors": errors,
    }
    logger.info(f"Google Trends collection complete: {result}")
    return result
