"""
data_collectors/data_processor.py

Module: Data Processing Layer. Takes the heterogeneous raw documents
sitting in MongoDB `raw_content` (one shape per platform: YouTube,
Reddit, Google Trends) and normalizes them into a single
`processed_content` schema that every downstream module (similarity
analysis, ranking engine, content generator) can consume without caring
which platform something came from.

Responsibilities (per the proposal's "Data Processing Layer"):
  - Remove duplicate content
  - Filter irrelevant/low-signal content
  - Extract useful metadata
  - Analyze engagement metrics
  - Prepare content for AI processing (a clean `text_for_ai` field)
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from database.mongodb import find, get_collection, upsert
from utils.logger import logger

# Minimum engagement thresholds below which raw content is considered noise
# and filtered out before it ever reaches the (expensive) AI layer.
_MIN_ENGAGEMENT = {
    "youtube": {"view_count": 500},
    "reddit": {"score": 20},
    "google_trends": {},  # trend keywords have no engagement field of their own
}


def _normalize_youtube(doc: Dict) -> Dict:
    return {
        "platform": "youtube",
        "external_id": doc["external_id"],
        "title": doc.get("title", ""),
        "text_for_ai": f"{doc.get('title', '')}\n{doc.get('description', '')[:500]}",
        "channel_id": doc.get("channel_id", ""),
        "author": doc.get("channel_title", ""),
        "url": f"https://www.youtube.com/watch?v={doc['external_id']}",
        "published_at": doc.get("published_at", ""),
        "tags": doc.get("tags", []),
        "engagement": {
            "views": doc.get("view_count", 0),
            "likes": doc.get("like_count", 0),
            "comments": doc.get("comment_count", 0),
        },
        "source": doc.get("source", "youtube"),
    }


def _normalize_reddit(doc: Dict) -> Dict:
    return {
        "platform": "reddit",
        "external_id": doc["external_id"],
        "title": doc.get("title", ""),
        "text_for_ai": f"{doc.get('title', '')}\n{doc.get('selftext', '')[:500]}",
        "channel_id": doc.get("subreddit", ""),
        "author": doc.get("author", ""),
        "url": doc.get("permalink", ""),
        "published_at": doc.get("created_utc", ""),
        "tags": [doc.get("flair")] if doc.get("flair") else [],
        "engagement": {
            "score": doc.get("score", 0),
            "upvote_ratio": doc.get("upvote_ratio", 0),
            "comments": doc.get("num_comments", 0),
        },
        "source": "reddit",
    }


def _normalize_google_trends(doc: Dict) -> Dict:
    keyword = doc.get("keyword", "")
    return {
        "platform": "google_trends",
        "external_id": doc["external_id"],
        "title": keyword,
        "text_for_ai": keyword,
        "channel_id": "",
        "author": "",
        "url": f"https://trends.google.com/trends/explore?q={keyword.replace(' ', '+')}",
        "published_at": doc.get("collected_at", datetime.utcnow()).isoformat() if isinstance(doc.get("collected_at"), datetime) else "",
        "tags": [],
        "engagement": {"rank": doc.get("rank", 999)},
        "source": "google_trends",
    }


_NORMALIZERS = {
    "youtube": _normalize_youtube,
    "reddit": _normalize_reddit,
    "google_trends": _normalize_google_trends,
}


def _passes_filter(platform: str, engagement: Dict) -> bool:
    thresholds = _MIN_ENGAGEMENT.get(platform, {})
    for field, minimum in thresholds.items():
        if engagement.get(field, 0) < minimum:
            return False
    return True


def process_platform(platform: str, batch_size: int = 500) -> Dict[str, int]:
    """
    Processes all not-yet-processed raw_content docs for one platform:
    normalize -> filter -> dedupe (via upsert on external_id) -> store.
    """
    normalizer = _NORMALIZERS.get(platform)
    if not normalizer:
        raise ValueError(f"No normalizer registered for platform='{platform}'.")

    raw_docs = find("raw_content", {"platform": platform, "processed": {"$ne": True}}, limit=batch_size)
    stats = {"seen": len(raw_docs), "filtered_out": 0, "processed": 0}

    for raw in raw_docs:
        normalized = normalizer(raw)
        if not _passes_filter(platform, normalized["engagement"]):
            stats["filtered_out"] += 1
        else:
            normalized["processed_at"] = datetime.utcnow()
            upsert("processed_content", {"platform": platform, "external_id": normalized["external_id"]}, normalized)
            stats["processed"] += 1

        get_collection("raw_content").update_one({"_id": raw["_id"]}, {"$set": {"processed": True}})

    logger.info(f"Processed platform={platform}: {stats}")
    return stats


def process_all_platforms() -> Dict[str, Dict[str, int]]:
    """Runs the full Data Processing Layer across every collector's raw output."""
    results = {}
    for platform in _NORMALIZERS:
        results[platform] = process_platform(platform)
    return results


def get_processed_content(platform: Optional[str] = None, limit: int = 200) -> List[Dict]:
    query = {"platform": platform} if platform else {}
    return find("processed_content", query, limit=limit, sort=[("processed_at", -1)])
