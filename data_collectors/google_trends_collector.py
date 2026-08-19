"""
data_collectors/google_trends_collector.py

Module 1 (Google Trends part) — uses `pytrends` (unofficial but widely
used Google Trends client; no API key required) to pull:
  - currently trending search terms for a region
  - interest-over-time for configured keywords (to compute growth velocity)
  - related/rising queries for a keyword (to surface adjacent opportunities)
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from pytrends.request import TrendReq

from config.settings import settings
from database.mongodb import upsert
from utils.logger import logger

_pytrends_client = None


def get_trends_client() -> TrendReq:
    global _pytrends_client
    if _pytrends_client is None:
        _pytrends_client = TrendReq(hl="en-US", tz=360)
    return _pytrends_client


def _store_raw(doc: Dict) -> None:
    upsert("raw_content", {"platform": "google_trends", "external_id": doc["external_id"]}, {**doc, "platform": "google_trends", "collected_at": datetime.utcnow()})


def get_trending_searches(geo: str = None) -> List[Dict]:
    """Module 1: today's trending search terms for a region."""
    geo = geo or settings.google_trends_geo
    client = get_trends_client()
    try:
        df = client.trending_searches(pn=geo.lower() if geo else "united_states")
    except Exception as exc:  # noqa: BLE001 — pytrends raises generic exceptions on scrape failures
        logger.error(f"Google Trends trending_searches failed: {exc}")
        return []

    docs = []
    for rank, row in enumerate(df[0].tolist()):
        doc = {
            "external_id": f"{geo}:{datetime.utcnow().date().isoformat()}:{rank}:{row}",
            "keyword": row,
            "rank": rank,
            "geo": geo,
        }
        docs.append(doc)
        _store_raw(doc)

    logger.info(f"Collected {len(docs)} trending searches for geo={geo}.")
    return docs


def get_interest_over_time(keywords: List[str], timeframe: str = "now 7-d") -> Dict[str, List[Dict]]:
    """
    Module 1 + growth-velocity input for the ranking engine: pulls the
    interest-over-time series for up to 5 keywords at a time (a pytrends/
    Google Trends API limit).
    """
    client = get_trends_client()
    results: Dict[str, List[Dict]] = {}

    for i in range(0, len(keywords), 5):
        batch = keywords[i : i + 5]
        try:
            client.build_payload(batch, timeframe=timeframe, geo=settings.google_trends_geo)
            df = client.interest_over_time()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Google Trends interest_over_time failed for {batch}: {exc}")
            continue

        if df.empty:
            continue

        for kw in batch:
            if kw not in df.columns:
                continue
            series = [{"timestamp": ts.isoformat(), "value": int(val)} for ts, val in df[kw].items()]
            results[kw] = series
            _store_raw({
                "external_id": f"interest:{kw}:{timeframe}",
                "keyword": kw,
                "timeframe": timeframe,
                "series": series,
            })

    logger.info(f"Collected interest-over-time for {list(results.keys())}.")
    return results


def get_related_queries(keyword: str) -> Dict[str, List[Dict]]:
    """Module 1: rising/top related queries for a single keyword — good source of adjacent video ideas."""
    client = get_trends_client()
    try:
        client.build_payload([keyword], geo=settings.google_trends_geo)
        related = client.related_queries()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Google Trends related_queries failed for '{keyword}': {exc}")
        return {"top": [], "rising": []}

    data = related.get(keyword, {}) or {}
    top_df, rising_df = data.get("top"), data.get("rising")
    result = {
        "top": top_df.to_dict("records") if top_df is not None else [],
        "rising": rising_df.to_dict("records") if rising_df is not None else [],
    }
    _store_raw({"external_id": f"related:{keyword}", "keyword": keyword, **result})
    return result


def run_full_collection() -> Dict[str, int]:
    """Module 1: trending searches + interest-over-time + related queries for every configured keyword."""
    counts = {"trending_searches": 0, "keywords_tracked": 0}
    counts["trending_searches"] = len(get_trending_searches())

    if settings.google_trends_keywords:
        interest = get_interest_over_time(settings.google_trends_keywords)
        counts["keywords_tracked"] = len(interest)
        for kw in settings.google_trends_keywords:
            get_related_queries(kw)
    else:
        logger.warning("GOOGLE_TRENDS_KEYWORDS is empty — skipping interest-over-time/related queries.")

    logger.info(f"Google Trends collection complete: {counts}")
    return counts
