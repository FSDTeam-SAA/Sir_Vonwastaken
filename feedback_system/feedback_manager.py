"""
feedback_system/feedback_manager.py

Module 6 — Learning Feedback System.

Collects and analyzes user feedback on recommendations to improve future
recommendation ranking. Tracks accept/reject/ignore actions and updates
recommendation weights based on patterns.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from bson import ObjectId
from database.mongodb import find, find_one, get_collection, insert_one, upsert
from utils.logger import logger


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
    if action not in ["accept", "reject", "ignore"]:
        raise ValueError(f"Invalid action: {action}. Must be 'accept', 'reject', or 'ignore'")
    
    feedback_doc = {
        "trend_id": trend_id,
        "creator_id": creator_id,
        "action": action,
        "notes": notes,
        "created_at": datetime.utcnow(),
    }
    
    result = insert_one("trend_feedback", feedback_doc)
    logger.info(f"Recorded {action} feedback for trend {trend_id} by {creator_id}")
    
    return feedback_doc


def get_feedback_history(trend_id: Optional[str] = None, creator_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
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
        query["trend_id"] = trend_id
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
        "total_feedback": len(feedback),
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
        summary[f"{action}_count"] += 1
        
        # Track categories
        trend = find_one("trend_candidates", {"_id": ObjectId(fb.get("trend_id"))})
        if trend:
            category = trend.get("analysis", {}).get("category", "unknown")
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
    feedback_summary = get_feedback_summary(creator_id)
    
    # Current weights from config (as baseline)
    from config.settings import settings
    current_weights = {
        "growth": settings.trend_weight_growth,
        "engagement": settings.trend_weight_engagement,
        "freshness": settings.trend_weight_freshness,
        "similarity": settings.trend_weight_similarity,
        "cross_platform": settings.trend_weight_cross_platform,
    }
    
    # Adjust similarity weight based on accept rate (if creator likes similar content)
    accept_rate = feedback_summary.get("accept_rate", 0.5)
    new_similarity = current_weights["similarity"] * (1 - alpha) + (accept_rate * current_weights["similarity"] * 2) * alpha
    
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
