"""
email_assistant/detect_sponsorship.py

Module 7 — Brand Deal Email Assistant (detection step).

Identifies which inbox messages are sponsorship/partnership/collaboration
opportunities. Uses a fast keyword pre-filter (cheap, catches obvious
cases) followed by an LLM classification call for anything ambiguous —
mirrors the proposal's cost-optimization principle ("rule-based filtering
will remove low-quality content; AI models will only analyze high-
potential candidates").
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from database.mongodb import find, get_collection
from utils.llm_client import chat_complete_json
from utils.logger import logger

_KEYWORDS = [
    "sponsor", "sponsorship", "partnership", "collaboration", "collab",
    "paid promotion", "brand deal", "campaign", "ambassador", "affiliate",
    "product placement", "influencer", "advertise", "advertising opportunity",
]

_SYSTEM_PROMPT = (
    "You classify inbound emails to a content creator as either a genuine business "
    "sponsorship/partnership/collaboration opportunity, or something else (fan mail, "
    "spam, newsletters, personal correspondence, platform notifications)."
)


def _keyword_hit(subject: str, body: str) -> bool:
    text = f"{subject} {body}".lower()
    return any(kw in text for kw in _KEYWORDS)


def classify_email(subject: str, body: str) -> Dict:
    """
    Returns {"is_sponsorship": bool, "confidence": float, "reason": str}.
    Only calls the LLM when the cheap keyword filter finds a signal,
    to avoid burning API calls on obviously irrelevant mail.
    """
    if not _keyword_hit(subject, body):
        return {"is_sponsorship": False, "confidence": 0.95, "reason": "No sponsorship-related keywords found."}

    prompt = f"""Email subject: {subject}
Email body (truncated):
\"\"\"{body[:1500]}\"\"\"

Is this a genuine sponsorship / brand partnership / paid collaboration opportunity for a content creator?
Respond with a JSON object: {{"is_sponsorship": true/false, "confidence": 0.0-1.0, "reason": "short reason"}}
Output ONLY the JSON, no prose."""

    result = chat_complete_json(_SYSTEM_PROMPT, prompt, max_tokens=150, temperature=0.2)
    if not result:
        return {"is_sponsorship": True, "confidence": 0.5, "reason": "Keyword match; LLM classification unavailable."}
    return {
        "is_sponsorship": bool(result.get("is_sponsorship", False)),
        "confidence": float(result.get("confidence", 0.5)),
        "reason": result.get("reason", ""),
    }


def scan_inbox_for_sponsorships(limit: int = 100) -> List[Dict]:
    """
    Module 7 batch runner: classifies every email in MongoDB `emails`
    (populated by data_collectors.gmail_collector) that hasn't been
    classified yet, and flags sponsorship ones for the summarize/draft
    steps.
    """
    docs = find("emails", {"sponsorship_classification": {"$exists": False}}, limit=limit)
    sponsorship_emails: List[Dict] = []

    for doc in docs:
        classification = classify_email(doc.get("subject", ""), doc.get("body", ""))
        get_collection("emails").update_one(
            {"_id": doc["_id"]},
            {"$set": {"sponsorship_classification": classification, "classified_at": datetime.utcnow()}},
        )
        if classification["is_sponsorship"]:
            sponsorship_emails.append({**doc, "sponsorship_classification": classification})

    logger.info(f"Scanned {len(docs)} emails; found {len(sponsorship_emails)} sponsorship candidates.")
    return sponsorship_emails


def get_sponsorship_emails(limit: int = 50) -> List[Dict]:
    return find("emails", {"sponsorship_classification.is_sponsorship": True}, limit=limit, sort=[("collected_at", -1)])
