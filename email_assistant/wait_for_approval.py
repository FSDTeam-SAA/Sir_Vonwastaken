"""
email_assistant/wait_for_approval.py

Module 7 — Brand Deal Email Assistant (human-approval workflow).

Nothing here auto-sends anything. AI-drafted replies sit in MongoDB
`email_drafts` with status="pending_approval" (and as a real Gmail
draft) until the creator explicitly approves or rejects them — via the
API endpoints in api/routes.py (POST /emails/drafts/{id}/approve|reject).

This is intentionally a request/response workflow rather than a blocking
`sleep`-based "wait" — appropriate for a FastAPI service where the
creator approves from the dashboard (Module 8) whenever they get to it.
"""
from __future__ import annotations

from bson import ObjectId
from bson.errors import InvalidId
from datetime import datetime
from typing import Dict, List, Optional

from data_collectors.gmail_collector import send_draft
from database.mongodb import find, find_one, get_collection
from utils.logger import logger


def _to_object_id(draft_id: str) -> ObjectId:
    try:
        return ObjectId(draft_id)
    except InvalidId as exc:
        raise ValueError(f"Invalid draft_id format: {draft_id}") from exc


def get_pending_drafts(limit: int = 50) -> List[Dict]:
    return find("email_drafts", {"status": "pending_approval"}, limit=limit, sort=[("created_at", -1)])


def get_draft(draft_id: str) -> Optional[Dict]:
    return find_one("email_drafts", {"_id": _to_object_id(draft_id)})


def approve_draft(draft_id: str) -> Dict:
    """
    Human approval step: sends the Gmail draft that was already created
    by email_assistant.draft_replies, and marks it approved+sent.
    """
    draft = get_draft(draft_id)
    if not draft:
        raise ValueError(f"No draft found with id={draft_id}.")
    if draft["status"] != "pending_approval":
        raise ValueError(f"Draft {draft_id} is not pending approval (status={draft['status']}).")

    sent_ok = False
    if draft.get("gmail_draft_id"):
        sent_ok = send_draft(draft["gmail_draft_id"])

    new_status = "sent" if sent_ok else "approved_send_failed"
    get_collection("email_drafts").update_one(
        {"_id": draft["_id"]}, {"$set": {"status": new_status, "approved_at": datetime.utcnow()}}
    )
    logger.info(f"Draft {draft_id} approved -> status={new_status}.")
    return {**draft, "status": new_status}


def reject_draft(draft_id: str, reason: str = "") -> Dict:
    draft = get_draft(draft_id)
    if not draft:
        raise ValueError(f"No draft found with id={draft_id}.")

    get_collection("email_drafts").update_one(
        {"_id": draft["_id"]},
        {"$set": {"status": "rejected", "rejection_reason": reason, "rejected_at": datetime.utcnow()}},
    )
    logger.info(f"Draft {draft_id} rejected. Reason: {reason or '(none given)'}")
    return {**draft, "status": "rejected", "rejection_reason": reason}
