"""
feedback_system/feedback_manager.py

Module 6 — Learning Feedback System.

Collects and analyzes user feedback on recommendations to improve future
recommendation ranking. Tracks accept/reject/ignore actions and updates
recommendation weights based on patterns.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId
from database.mongodb import find, find_one, insert_one, upsert
from utils.logger import logger


_VALID_ACTIONS = frozenset({"accept", "reject", "ignore"})


def _as_object_id(value: Any) -> Optional[ObjectId]:
    """Return an ObjectId only when *value* is a valid MongoDB identifier.

    Feedback IDs are normally external content IDs (for example a YouTube
    video ID), so constructing ``ObjectId(value)`` unconditionally makes the
    feedback summary fail for valid API input.  This helper is deliberately
    conservative so legacy ObjectId-based feedback can still be read without
    treating all external IDs as database IDs.
    """
    if isinstance(value, ObjectId):
        return value
    if isinstance(value, str) and ObjectId.is_valid(value):
        return ObjectId(value)
    return None


def _find_trend_candidate(trend_id: Any, trend_object_id: Any = None) -> Optional[Dict]:
    """Find a ranked trend from an external ID, with a legacy ObjectId fallback."""
    if trend_id is None and trend_object_id is None:
        return None

    # ``trend_candidates.content_id`` is the stable public identifier exposed
    # by the trends and content-generation endpoints.
    if trend_id is not None:
        trend = find_one("trend_candidates", {"content_id": str(trend_id)})
        if trend:
            return trend

    # New feedback documents can retain the candidate's database ID for
    # traceability.  Older documents may instead have stored it in trend_id.
    object_id = _as_object_id(trend_object_id) or _as_object_id(trend_id)
    if object_id:
        return find_one("trend_candidates", {"_id": object_id})
    return None


def _resolve_trend_reference(trend_id: str) -> Tuple[str, Optional[str]]:
    """Return the canonical external content ID and optional MongoDB ID.

    The public API accepts a trend's external ``content_id``.  Accepting a
    valid legacy MongoDB ObjectId as well makes migrations non-breaking, but
    feedback is always stored against the canonical external ID when it can
    be resolved.
    """
    requested_id = str(trend_id)
    trend = _find_trend_candidate(requested_id)
    if not trend:
        return requested_id, None

    content_id = trend.get("content_id")
    canonical_id = str(content_id) if content_id is not None else requested_id
    object_id = trend.get("_id")
    return canonical_id, str(object_id) if object_id is not None else None


def _trend_category(trend: Dict) -> str:
    """Read a trend category, joining processed content when ranking omitted it."""
    analysis = trend.get("analysis")
    analysis = analysis if isinstance(analysis, dict) else {}
    category = analysis.get("category") or trend.get("category")
    if category:
        return str(category)

    # Older candidates may not carry the AI analysis that newer ranked
    # candidates preserve. Resolve it here so feedback analytics still
    # produces meaningful category preferences for both shapes.
    content_id = trend.get("content_id")
    if content_id is not None:
        processed = find_one("processed_content", {"external_id": content_id})
        processed_analysis = (processed or {}).get("analysis")
        processed_analysis = processed_analysis if isinstance(processed_analysis, dict) else {}
        category = processed_analysis.get("category") or (processed or {}).get("category")
        if category:
            return str(category)

    return "unknown"


def record_feedback(trend_id: str, creator_id: str, action: str, notes: str = "") -> Dict:
    """
    Record user feedback on a trend recommendation.
    
    Args:
        trend_id: ID of the trend being rated
        creator_id: ID of the creator giving feedback
        action: One of "accept", "reject", "ignore"
        notes: Optional notes from the creator
    
    Returns:
        Feedback document stored in MongoDB
    """
    if not isinstance(action, str) or action not in _VALID_ACTIONS:
        raise ValueError(f"Invalid action: {action}. Must be 'accept', 'reject', or 'ignore'")

    canonical_trend_id, trend_object_id = _resolve_trend_reference(trend_id)
    feedback_doc = {
        "trend_id": canonical_trend_id,
        "creator_id": creator_id,
        "action": action,
        "notes": notes,
        "created_at": datetime.utcnow(),
    }
    if trend_object_id:
        feedback_doc["trend_object_id"] = trend_object_id

    # PyMongo adds an ObjectId to the document it receives.  Replace that
    # value with its string form before returning it through FastAPI.
    feedback_doc["_id"] = insert_one("trend_feedback", feedback_doc)
    logger.info(f"Recorded {action} feedback for trend {canonical_trend_id} by {creator_id}")

    return feedback_doc


def get_feedback_history(
    trend_id: Optional[str] = None,
    creator_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict]:
    """
    Retrieve feedback history.
    
    Args:
        trend_id: Filter by specific trend (optional)
        creator_id: Filter by specific creator (optional)
        limit: Maximum number of records to retrieve
    
    Returns:
        List of feedback documents
    """
    query = {}
    if trend_id:
        canonical_id, trend_object_id = _resolve_trend_reference(trend_id)
        matching_ids = list(dict.fromkeys(filter(None, [str(trend_id), canonical_id])))
        trend_filters = [{"trend_id": value} for value in matching_ids]
        legacy_requested_id = _as_object_id(trend_id)
        if legacy_requested_id:
            trend_filters.append({"trend_id": legacy_requested_id})
        if trend_object_id:
            trend_filters.append({"trend_object_id": trend_object_id})
            legacy_object_id = _as_object_id(trend_object_id)
            if legacy_object_id:
                trend_filters.append({"trend_object_id": legacy_object_id})
        query["$or"] = trend_filters
    if creator_id:
        query["creator_id"] = creator_id
    
    return find("trend_feedback", query, limit=limit, sort=[("created_at", -1)])


def get_feedback_summary(creator_id: str, days: int = 30) -> Dict:
    """
    Get summary of feedback for a creator over last N days.
    
    Args:
        creator_id: Creator ID
        days: Number of days to analyze
    
    Returns:
        Summary dict with accept/reject/ignore counts and trending patterns
    """
    from datetime import timedelta
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    feedback = find(
        "trend_feedback",
        {
            "creator_id": creator_id,
            "created_at": {"$gte": cutoff_date}
        }
    )
    
    summary = {
        "period_days": days,
        "total_feedback": 0,
        "accept_count": 0,
        "reject_count": 0,
        "ignore_count": 0,
        "accept_rate": 0.0,
        "top_accepted_categories": [],
        "top_rejected_categories": [],
    }
    
    category_accept = {}
    category_reject = {}
    
    for fb in feedback:
        action = fb.get("action", "")
        if not isinstance(action, str) or action not in _VALID_ACTIONS:
            logger.warning(f"Ignoring feedback with invalid action '{action}'")
            continue
        summary["total_feedback"] += 1
        summary[f"{action}_count"] += 1
        
        # Track categories
        trend = _find_trend_candidate(fb.get("trend_id"), fb.get("trend_object_id"))
        if trend:
            category = _trend_category(trend)
            if action == "accept":
                category_accept[category] = category_accept.get(category, 0) + 1
            elif action == "reject":
                category_reject[category] = category_reject.get(category, 0) + 1
    
    if summary["total_feedback"] > 0:
        summary["accept_rate"] = summary["accept_count"] / summary["total_feedback"]
        summary["top_accepted_categories"] = sorted(
            category_accept.items(), key=lambda x: x[1], reverse=True
        )[:5]
        summary["top_rejected_categories"] = sorted(
            category_reject.items(), key=lambda x: x[1], reverse=True
        )[:5]
    
    return summary


def update_recommendation_weights(creator_id: str, alpha: float = 0.1) -> Dict:
    """
    Update trend ranking weights based on user feedback.
    
    Uses exponential moving average to adjust weights without erasing history.
    
    Args:
        creator_id: Creator ID
        alpha: Learning rate (0-1, default 0.1). Higher = more responsive to recent feedback
    
    Returns:
        New weights dict
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be greater than 0 and no more than 1")

    feedback_summary = get_feedback_summary(creator_id)
    
    # Current weights from config (as baseline)
    from config.settings import settings
    default_weights = {
        "growth": settings.trend_weight_growth,
        "engagement": settings.trend_weight_engagement,
        "freshness": settings.trend_weight_freshness,
        "similarity": settings.trend_weight_similarity,
        "cross_platform": settings.trend_weight_cross_platform,
    }

    stored_weights = get_personalized_weights(creator_id)
    current_weights = dict(default_weights)
    if isinstance(stored_weights, dict):
        for name, default_value in default_weights.items():
            value = stored_weights.get(name, default_value)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value >= 0
            ):
                current_weights[name] = float(value)

    # With no observations there is nothing to learn. Returning the current
    # values avoids silently lowering similarity from its configured default.
    if feedback_summary["total_feedback"] == 0:
        logger.info(
            f"No feedback available for creator {creator_id}; recommendation weights unchanged."
        )
        return current_weights
    
    # Adjust similarity weight based on accept rate (if creator likes similar content)
    accept_rate = feedback_summary["accept_rate"]
    new_similarity = current_weights["similarity"] * (1 - alpha) + (
        accept_rate * current_weights["similarity"] * 2
    ) * alpha
    
    # Adjust based on top categories
    # If creator accepts certain categories, boost their signals
    personalized_weights = {
        "growth": current_weights["growth"],
        "engagement": current_weights["engagement"],
        "freshness": current_weights["freshness"],
        "similarity": min(new_similarity, 1.0),  # Cap at 1.0
        "cross_platform": current_weights["cross_platform"],
    }
    
    # Store updated weights for this creator
    upsert(
        "creator_weights",
        {"creator_id": creator_id},
        {
            "creator_id": creator_id,
            "weights": personalized_weights,
            "updated_at": datetime.utcnow(),
            "feedback_count": feedback_summary["total_feedback"],
        },
    )
    
    logger.info(f"Updated recommendation weights for creator {creator_id}")
    return personalized_weights


def get_personalized_weights(creator_id: str) -> Optional[Dict]:
    """
    Get personalized recommendation weights for a creator.
    
    Returns stored weights if available, None otherwise.
    """
    doc = find_one("creator_weights", {"creator_id": creator_id})
    return doc.get("weights") if doc else None
