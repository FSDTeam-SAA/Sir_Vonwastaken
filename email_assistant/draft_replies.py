"""
email_assistant/draft_replies.py

Module 7/8 — Brand Deal Email Assistant (draft-generation step).

Generates a professional reply to a sponsorship email, saves it as an
actual Gmail draft (visible in the creator's Gmail "Drafts" folder), and
records it in MongoDB `email_drafts` with status="pending_approval" —
nothing is ever sent from here. Sending only happens from
email_assistant.wait_for_approval, after explicit human approval
(Module 7's "wait for user approval before sending any communication").
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from data_collectors.gmail_collector import create_draft
from database.mongodb import find_one, insert_one
from utils.llm_client import chat_complete
from utils.logger import logger

_SYSTEM_PROMPT = (
    "You are a professional, friendly assistant drafting email replies on behalf of a "
    "content creator responding to sponsorship/brand-partnership inquiries. Keep replies "
    "concise, warm, and business-appropriate. Never commit to specific pricing or dates "
    "unless the creator's notes explicitly provide them."
)


def generate_reply_text(email_subject: str, email_body: str, summary: Dict, creator_notes: str = "") -> str:
    notes_block = f"\nCreator's notes/instructions for this reply: {creator_notes}" if creator_notes else ""
    prompt = f"""Original email subject: {email_subject}
Original email summary: {summary.get('summary', '')}
Brand: {summary.get('brand_name', 'the brand')}
Their offer/request: {summary.get('offer_details', '')}{notes_block}

Write a professional reply email (not a subject line, just the body) that:
- Thanks them for reaching out
- Shows genuine interest
- Asks for any missing key details needed to move forward (e.g. rate card, timeline, deliverables) if not already provided
- Ends with a clear, friendly next step
- Is 3-6 short paragraphs, plain text, no markdown

Write the reply now."""

    return chat_complete(_SYSTEM_PROMPT, prompt, max_tokens=400, temperature=0.6).strip()


def create_draft_reply(email_external_id: str, creator_notes: str = "") -> Dict:
    """
    Module 7/8: builds the reply text, creates a real Gmail draft, and
    stores a pending-approval record. Returns the stored draft document.
    """
    email_doc = find_one("emails", {"external_id": email_external_id})
    if not email_doc:
        raise ValueError(f"No email found with external_id={email_external_id}.")

    summary = email_doc.get("summary")
    if not summary:
        raise ValueError(
            f"Email external_id={email_external_id} has not been summarized yet. "
            "Call email_assistant.summarize_email.summarize_and_store first."
        )

    reply_text = generate_reply_text(email_doc.get("subject", ""), email_doc.get("body", ""), summary, creator_notes)

    reply_to = email_doc.get("from", "")
    subject = f"Re: {email_doc.get('subject', '')}"
    gmail_draft_id = create_draft(to=reply_to, subject=subject, body=reply_text, thread_id=email_doc.get("thread_id"))

    draft_doc = {
        "email_external_id": email_external_id,
        "gmail_draft_id": gmail_draft_id,
        "to": reply_to,
        "subject": subject,
        "body": reply_text,
        "status": "pending_approval",
        "created_at": datetime.utcnow(),
    }
    draft_id = insert_one("email_drafts", draft_doc)
    draft_doc["_id"] = draft_id

    logger.info(f"Created draft reply for email={email_external_id} (gmail_draft_id={gmail_draft_id}), awaiting approval.")
    return draft_doc
