"""
email_assistant/summarize_email.py

Module 7 — Brand Deal Email Assistant (summarization step).

For emails flagged as sponsorship opportunities, extracts:
  - a short summary
  - brand name / company
  - what's being offered / requested
  - any deadline or budget mentioned

Stored back onto the email document in MongoDB so the dashboard (Module
8) and draft-reply step can use it without re-calling the LLM.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict

from database.mongodb import find_one, get_collection
from utils.llm_client import chat_complete_json
from utils.logger import logger

_SYSTEM_PROMPT = (
    "You extract structured information from sponsorship/brand-partnership emails sent "
    "to a content creator, so the creator can quickly triage opportunities."
)


def summarize_email(subject: str, body: str) -> Dict:
    """
    Returns:
        {"summary": str, "brand_name": str, "offer_details": str,
         "deadline": str, "budget_mentioned": str}
    """
    prompt = f"""Email subject: {subject}
Email body:
\"\"\"{body[:2500]}\"\"\"

Extract the following as a JSON object:
- "summary": one to two sentence summary of the email
- "brand_name": the company/brand name (empty string if unclear)
- "offer_details": what they are proposing/requesting
- "deadline": any date/deadline mentioned (empty string if none)
- "budget_mentioned": any budget/payment figure mentioned (empty string if none)

Output ONLY the JSON, no prose."""

    result = chat_complete_json(_SYSTEM_PROMPT, prompt, max_tokens=300, temperature=0.3)
    return {
        "summary": result.get("summary", ""),
        "brand_name": result.get("brand_name", ""),
        "offer_details": result.get("offer_details", ""),
        "deadline": result.get("deadline", ""),
        "budget_mentioned": result.get("budget_mentioned", ""),
    }


def summarize_and_store(email_external_id: str) -> Dict:
    email_doc = find_one("emails", {"external_id": email_external_id})
    if not email_doc:
        raise ValueError(f"No email found with external_id={email_external_id}. Sync Gmail first.")

    summary = summarize_email(email_doc.get("subject", ""), email_doc.get("body", ""))
    get_collection("emails").update_one(
        {"external_id": email_external_id},
        {"$set": {"summary": summary, "summarized_at": datetime.utcnow()}},
    )
    logger.info(f"Summarized email external_id={email_external_id} from brand='{summary.get('brand_name')}'.")
    return summary
