"""
trend_ranking/ranking_engine.py

Module 4 — Trend Scoring Engine.

Combines several signals into a single 0-1 trend score per piece of
processed content:
  - growth velocity   (views/score accumulated per hour since publish)
  - engagement rate    (likes/comments relative to views, or upvote ratio)
  - freshness           (exponential decay by content age)
  - creator similarity (Module 3's embedding similarity)
  - cross-platform presence (same topic showing up on >1 platform)

Weights are configurable via .env (TREND_WEIGHT_*, see config/settings.py)
so the creator can tune what matters most without touching code.

Only items scoring at/above TREND_NOTIFY_SCORE_THRESHOLD trigger a
notification (Module: Notification Layer) and get written to
`trend_candidates`, which is what AI_generator.generate_content reads
from to produce the actual titles/hooks/scripts.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config.settings import settings
from content_similarity_check.embedding_search import get_content_embedding
from content_similarity_check.vector_search import score_content_against_creator
from database.mongodb import find, get_collection, strip_ids, upsert
from notification_system import desktop_notifications, discord, email as email_notify, telegram
from utils.logger import logger


def _parse_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        # Handles both YouTube's RFC3339 and Reddit's ISO format
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _age_hours(published_at: str) -> float:
    dt = _parse_dt(published_at)
    if not dt:
        return 999.0
    delta = datetime.now(timezone.utc) - dt
    return max(delta.total_seconds() / 3600.0, 0.01)


def freshness_score(published_at: str, half_life_hours: float = 24.0) -> float:
    """Exponential decay — content published `half_life_hours` ago scores 0.5."""
    age = _age_hours(published_at)
    return math.exp(-math.log(2) * age / half_life_hours)


def growth_velocity_score(doc: Dict) -> float:
    """Normalized (0-1, log-scaled) rate of accumulation per hour since publish."""
    age = _age_hours(doc.get("published_at", ""))
    engagement = doc.get("engagement", {})

    if doc["platform"] == "youtube":
        raw_rate = engagement.get("views", 0) / age
        # log-scale: ~100k views/hour saturates near 1.0
        return min(math.log1p(raw_rate) / math.log1p(100_000), 1.0)
    if doc["platform"] == "reddit":
        raw_rate = engagement.get("score", 0) / age
        return min(math.log1p(raw_rate) / math.log1p(2_000), 1.0)
    if doc["platform"] == "google_trends":
        rank = engagement.get("rank", 999)
        return max(1.0 - (rank / 25.0), 0.0)  # rank 0 (top trend) -> 1.0
    return 0.0


def engagement_rate_score(doc: Dict) -> float:
    engagement = doc.get("engagement", {})
    if doc["platform"] == "youtube":
        views = engagement.get("views", 0) or 1
        likes = engagement.get("likes", 0)
        comments = engagement.get("comments", 0)
        rate = (likes + comments) / views
        return min(rate * 20, 1.0)  # ~5% like+comment rate saturates at 1.0
    if doc["platform"] == "reddit":
        return float(engagement.get("upvote_ratio", 0) or 0)
    if doc["platform"] == "google_trends":
        return 0.5  # trends have no direct engagement signal of their own
    return 0.0


def cross_platform_score(doc: Dict, all_docs: List[Dict]) -> float:
    """
    Rough cross-platform-presence heuristic: counts how many OTHER
    processed_content docs (on a different platform) share at least one
    significant word with this doc's title. More shared-topic platforms
    = stronger signal the trend is broad, not a single-platform blip.
    """
    title_words = {w.lower() for w in doc.get("title", "").split() if len(w) > 3}
    if not title_words:
        return 0.0

    other_platforms = set()
    for other in all_docs:
        if other["platform"] == doc["platform"]:
            continue
        other_words = {w.lower() for w in other.get("title", "").split() if len(w) > 3}
        if title_words & other_words:
            other_platforms.add(other["platform"])

    return min(len(other_platforms) / 2.0, 1.0)  # 2+ other platforms = max score


def similarity_to_creator_score(doc: Dict, channel_id: Optional[str]) -> float:
    if not channel_id:
        return 0.0
    content_embedding = get_content_embedding(doc["external_id"])
    if not content_embedding:
        return 0.0
    raw_score = score_content_against_creator(content_embedding, channel_id)
    return max(min((raw_score + 1) / 2, 1.0), 0.0)  # cosine [-1,1] -> [0,1]


def score_trend(doc: Dict, channel_id: Optional[str], all_docs: List[Dict]) -> Dict:
    growth = growth_velocity_score(doc)
    engagement = engagement_rate_score(doc)
    freshness = freshness_score(doc.get("published_at", ""))
    similarity = similarity_to_creator_score(doc, channel_id)
    cross_platform = cross_platform_score(doc, all_docs)

    final_score = (
        growth * settings.trend_weight_growth
        + engagement * settings.trend_weight_engagement
        + freshness * settings.trend_weight_freshness
        + similarity * settings.trend_weight_similarity
        + cross_platform * settings.trend_weight_cross_platform
    )

    return {
        "score": round(final_score, 4),
        "breakdown": {
            "growth_velocity": round(growth, 4),
            "engagement_rate": round(engagement, 4),
            "freshness": round(freshness, 4),
            "creator_similarity": round(similarity, 4),
            "cross_platform": round(cross_platform, 4),
        },
    }


def rank_trends(channel_id: Optional[str] = None, limit: int = 300) -> List[Dict]:
    """
    Module 4: scores every processed_content item, writes results to
    `trend_candidates`, and fires a notification (Module: Notification
    Layer) for anything crossing TREND_NOTIFY_SCORE_THRESHOLD.
    """
    all_docs = find("processed_content", limit=limit, sort=[("processed_at", -1)])
    if not all_docs:
        logger.warning("rank_trends: no processed_content available. Run collection + processing first.")
        return []

    ranked: List[Dict] = []
    for doc in all_docs:
        scoring = score_trend(doc, channel_id, all_docs)
        candidate = {
            "content_id": doc["external_id"],
            "channel_id": channel_id,
            "platform": doc["platform"],
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "score": scoring["score"],
            "breakdown": scoring["breakdown"],
            "ranked_at": datetime.utcnow(),
        }
        upsert("trend_candidates", {"content_id": candidate["content_id"]}, candidate)
        ranked.append(candidate)

        if candidate["score"] >= settings.trend_notify_score_threshold:
            _notify_high_value_trend(candidate)

    ranked.sort(key=lambda c: c["score"], reverse=True)
    logger.info(f"Ranked {len(ranked)} trend candidates; "
                f"{sum(1 for c in ranked if c['score'] >= settings.trend_notify_score_threshold)} crossed the notify threshold.")
    return ranked


def _notify_high_value_trend(candidate: Dict) -> None:
    title = f"🔥 High-potential trend: {candidate['title'][:80]}"
    message = f"Platform: {candidate['platform']} | Score: {candidate['score']} | {candidate.get('url', '')}"

    if settings.notify_desktop_enabled:
        desktop_notifications.send(title, message)
    if settings.notify_discord_enabled:
        discord.send(title, message)
    if settings.notify_telegram_enabled:
        telegram.send(f"{title}\n{message}")
    if settings.notify_email_enabled:
        email_notify.send(title, message)


def get_top_trends(channel_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
    query = {"channel_id": channel_id} if channel_id else {}
    docs = find("trend_candidates", query, limit=limit, sort=[("score", -1)])
    return strip_ids(docs)
