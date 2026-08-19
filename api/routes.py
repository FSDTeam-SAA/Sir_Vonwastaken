"""
api/routes.py

All FastAPI endpoints for the platform, wired to the modules implemented
across data_collectors/, creator_profile/, content_similarity_check/,
trend_ranking/, AI_generator/, email_assistant/, and notification_system/.

Mounted in main.py under the "/api" prefix.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from AI_analysis.content_analyzer import analyze_pending_batch
from AI_generator.generate_content import (
    generate_content_for_trend,
    get_generated_content_history,
    regenerate_field,
)
from content_similarity_check.embedding_search import embed_pending_content
from content_similarity_check.vector_search import find_content_similar_to_creator
from creator_profile.build_creator_profile import (
    build_creator_profile_from_channel,
    get_creator_profile_summary,
)
from data_collectors import (
    data_processor,
    gmail_collector,
    google_trends_collector,
    reddit_collector,
    youtube_collector,
)
from database.mongodb import ping
from email_assistant.detect_sponsorship import get_sponsorship_emails, scan_inbox_for_sponsorships
from email_assistant.draft_replies import create_draft_reply
from email_assistant.summarize_email import summarize_and_store
from email_assistant.wait_for_approval import approve_draft, get_pending_drafts, reject_draft
from notification_system import desktop_notifications, discord, email as email_notify, telegram
from trend_ranking.ranking_engine import get_top_trends, rank_trends
from utils.logger import logger

router = APIRouter()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health")
def health_check():
    return {"status": "ok", "mongodb_connected": ping()}    


# ---------------------------------------------------------------------------
# Module 1 — Data Collection
# ---------------------------------------------------------------------------

@router.post("/collect/youtube")
def collect_youtube():
    try:
        return youtube_collector.run_full_collection()
    except Exception as exc:  # noqa: BLE001
        logger.exception("YouTube collection failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/collect/youtube/search")
def collect_youtube_search(query: str = Query(...), max_results: int = Query(25, le=50)):
    try:
        return {"videos": youtube_collector.search_videos(query, max_results=max_results)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/collect/reddit")
def collect_reddit():
    try:
        return reddit_collector.run_full_collection()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Reddit collection failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/collect/google-trends")
def collect_google_trends():
    try:
        return google_trends_collector.run_full_collection()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Google Trends collection failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/collect/gmail/sync")
def collect_gmail(max_results: int = Query(20, le=100), query: Optional[str] = None):
    try:
        messages = gmail_collector.list_recent_messages(max_results=max_results, query=query)
        return {"synced": len(messages)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Gmail sync failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/collect/all")
def collect_all():
    """Runs every Module-1 collector in sequence. Failures in one platform don't block the others."""
    results = {}
    for name, fn in [
        ("youtube", youtube_collector.run_full_collection),
        ("reddit", reddit_collector.run_full_collection),
        ("google_trends", google_trends_collector.run_full_collection),
    ]:
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Collector '{name}' failed: {exc}")
            results[name] = {"error": str(exc)}
    return results


# ---------------------------------------------------------------------------
# Module: Data Processing Layer
# ---------------------------------------------------------------------------

@router.post("/process/run")
def process_run():
    try:
        return data_processor.process_all_platforms()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Data processing failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Module 2 — Creator Profile
# ---------------------------------------------------------------------------

@router.post("/creator-profile/{channel_id}/build")
def build_profile(channel_id: str, max_videos: int = Query(25, le=50)):
    try:
        embedding = build_creator_profile_from_channel(channel_id, max_videos=max_videos)
        if embedding is None:
            raise HTTPException(status_code=404, detail=f"Could not build profile for channel_id={channel_id}: no videos found.")
        return get_creator_profile_summary(channel_id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Creator profile build failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/creator-profile/{channel_id}")
def get_profile(channel_id: str):
    profile = get_creator_profile_summary(channel_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"No creator profile found for channel_id={channel_id}.")
    return profile


# ---------------------------------------------------------------------------
# Module 3 — Similarity Analysis
# ---------------------------------------------------------------------------

@router.post("/similarity/embed-pending")
def embed_pending(limit: int = Query(200, le=1000)):
    try:
        return {"embedded": embed_pending_content(limit=limit)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/similarity/{channel_id}/matches")
def similarity_matches(channel_id: str, top_k: int = Query(20, le=100)):
    results = find_content_similar_to_creator(channel_id, top_k=top_k)
    return [{"content_id": ref_id, "score": score, "metadata": metadata} for ref_id, score, metadata in results]


# ---------------------------------------------------------------------------
# AI Intelligence Layer (categorization)
# ---------------------------------------------------------------------------

@router.post("/analysis/run")
def run_analysis(limit: int = Query(100, le=500)):
    try:
        return {"analyzed": analyze_pending_batch(limit=limit)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Module 4 — Trend Scoring & Ranking
# ---------------------------------------------------------------------------

@router.post("/trends/rank")
def trends_rank(channel_id: Optional[str] = None, limit: int = Query(300, le=1000)):
    try:
        return rank_trends(channel_id=channel_id, limit=limit)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Trend ranking failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/trends")
def trends_list(channel_id: Optional[str] = None, limit: int = Query(20, le=200)):
    return get_top_trends(channel_id=channel_id, limit=limit)


# ---------------------------------------------------------------------------
# Module 5 — AI Content Generation
# ---------------------------------------------------------------------------

class RegenerateFieldRequest(BaseModel):
    field: str
    channel_id: Optional[str] = None


@router.post("/content/generate/{content_id}")
def content_generate(content_id: str, channel_id: Optional[str] = None):
    try:
        return generate_content_for_trend(content_id, channel_id=channel_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Content generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/content/regenerate/{content_id}")
def content_regenerate(content_id: str, payload: RegenerateFieldRequest):
    try:
        return regenerate_field(content_id, payload.field, channel_id=payload.channel_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Content regeneration failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/content/history/{channel_id}")
def content_history(channel_id: str, limit: int = Query(20, le=100)):
    docs = get_generated_content_history(channel_id, limit=limit)
    for doc in docs:
        doc["_id"] = str(doc["_id"])
    return docs


# ---------------------------------------------------------------------------
# Module 7/8 — Brand Deal Email Assistant
# ---------------------------------------------------------------------------

@router.post("/emails/scan-sponsorships")
def emails_scan(limit: int = Query(100, le=500)):
    try:
        results = scan_inbox_for_sponsorships(limit=limit)
        for doc in results:
            doc["_id"] = str(doc["_id"])
        return results
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sponsorship scan failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/emails/sponsorships")
def emails_sponsorships(limit: int = Query(50, le=200)):
    docs = get_sponsorship_emails(limit=limit)
    for doc in docs:
        doc["_id"] = str(doc["_id"])
    return docs


@router.post("/emails/{email_external_id}/summarize")
def emails_summarize(email_external_id: str):
    try:
        return summarize_and_store(email_external_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


class DraftReplyRequest(BaseModel):
    creator_notes: str = ""


@router.post("/emails/{email_external_id}/draft-reply")
def emails_draft_reply(email_external_id: str, payload: DraftReplyRequest = DraftReplyRequest()):
    try:
        draft = create_draft_reply(email_external_id, creator_notes=payload.creator_notes)
        draft["_id"] = str(draft["_id"])
        return draft
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Draft reply creation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/emails/drafts")
def emails_drafts_list():
    docs = get_pending_drafts()
    for doc in docs:
        doc["_id"] = str(doc["_id"])
    return docs


@router.post("/emails/drafts/{draft_id}/approve")
def emails_draft_approve(draft_id: str):
    try:
        draft = approve_draft(draft_id)
        draft["_id"] = str(draft["_id"])
        return draft
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


class RejectDraftRequest(BaseModel):
    reason: str = ""


@router.post("/emails/drafts/{draft_id}/reject")
def emails_draft_reject(draft_id: str, payload: RejectDraftRequest = RejectDraftRequest()):
    try:
        draft = reject_draft(draft_id, reason=payload.reason)
        draft["_id"] = str(draft["_id"])
        return draft
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Notification Layer
# ---------------------------------------------------------------------------

class NotifyTestRequest(BaseModel):
    title: str = "Test notification"
    message: str = "This is a test notification from the Content Trend Intelligence Assistant."
    channels: List[str] = ["desktop"]


@router.post("/notify/test")
def notify_test(payload: NotifyTestRequest):
    results = {}
    dispatch = {
        "desktop": lambda: desktop_notifications.send(payload.title, payload.message),
        "discord": lambda: discord.send(payload.title, payload.message),
        "telegram": lambda: telegram.send(f"{payload.title}\n{payload.message}"),
        "email": lambda: email_notify.send(payload.title, payload.message),
    }
    for channel in payload.channels:
        fn = dispatch.get(channel)
        if not fn:
            results[channel] = "unknown channel"
            continue
        results[channel] = fn()
    return results


# ---------------------------------------------------------------------------
# Module 8 — Dashboard data
# ---------------------------------------------------------------------------

@router.get("/dashboard/{channel_id}")
def dashboard(channel_id: str):
    """Aggregated snapshot for the dashboard: profile, top trends, pending drafts, sponsorship emails."""
    return {
        "creator_profile": get_creator_profile_summary(channel_id),
        "top_trends": get_top_trends(channel_id=channel_id, limit=10),
        "pending_email_drafts": len(get_pending_drafts()),
        "sponsorship_emails": len(get_sponsorship_emails()),
        "mongodb_connected": ping(),
    }
