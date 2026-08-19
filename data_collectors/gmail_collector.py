"""
data_collectors/gmail_collector.py

Module 1 (Gmail part) + support for Module 7/8 (Brand Deal Email
Assistant). Uses the official Gmail API via OAuth2 (installed-app flow —
appropriate since this runs locally on the creator's own Mac, per the
proposal's deployment model).

First run opens a browser window for the creator to grant access, then
caches the refresh token in GMAIL_TOKEN_FILE so subsequent runs are
non-interactive. Scopes requested:
  - gmail.readonly  -> list/read messages for sponsorship detection
  - gmail.compose   -> create (but never send) draft replies, so the
                       human-approval step (Module 7) always sits between
                       the AI and an actually-sent email.
"""
from __future__ import annotations

import base64
import os
from datetime import datetime
from email.mime.text import MIMEText
from typing import Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config.settings import settings
from database.mongodb import upsert
from utils.logger import logger

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

_gmail_service = None


def get_gmail_service():
    """
    Returns an authenticated Gmail API service client, running the OAuth2
    installed-app flow on first use and reusing/refreshing the cached
    token afterwards.
    """
    global _gmail_service
    if _gmail_service is not None:
        return _gmail_service

    creds: Optional[Credentials] = None
    if os.path.exists(settings.gmail_token_file):
        creds = Credentials.from_authorized_user_file(settings.gmail_token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(settings.gmail_credentials_file):
                raise RuntimeError(
                    f"Gmail OAuth client secrets file not found at '{settings.gmail_credentials_file}'. "
                    "Download it from Google Cloud Console (OAuth client ID, Desktop app type) and set "
                    "GMAIL_CREDENTIALS_FILE in your .env to its path."
                )
            flow = InstalledAppFlow.from_client_secrets_file(settings.gmail_credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(settings.gmail_token_file, "w") as token_file:
            token_file.write(creds.to_json())

    _gmail_service = build("gmail", "v1", credentials=creds)
    return _gmail_service


def _extract_body(payload: Dict) -> str:
    """Walks a Gmail message payload to find the plain-text (falling back to HTML) body."""
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

    if payload.get("mimeType") == "text/html" and payload.get("body", {}).get("data") and not payload.get("parts"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

    for part in payload.get("parts", []) or []:
        body = _extract_body(part)
        if body:
            return body
    return ""


def _headers_dict(payload: Dict) -> Dict[str, str]:
    return {h["name"].lower(): h["value"] for h in payload.get("headers", [])}


def list_recent_messages(max_results: int = 20, query: Optional[str] = None) -> List[Dict]:
    """
    Module 1: lists recent inbox messages matching `query` (Gmail search
    syntax, e.g. "newer_than:2d"), fetches full content for each, and
    stores them into MongoDB `emails`.
    """
    service = get_gmail_service()
    query = query or settings.gmail_query
    try:
        list_resp = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    except HttpError as exc:
        logger.error(f"Gmail list failed: {exc}")
        return []

    message_ids = [m["id"] for m in list_resp.get("messages", [])]
    docs: List[Dict] = []
    for msg_id in message_ids:
        doc = get_message(msg_id)
        if doc:
            docs.append(doc)

    logger.info(f"Collected {len(docs)} Gmail messages for query='{query}'.")
    return docs


def get_message(message_id: str) -> Optional[Dict]:
    service = get_gmail_service()
    try:
        msg = service.users().messages().get(userId="me", id=message_id, format="full").execute()
    except HttpError as exc:
        logger.error(f"Gmail get message failed for id={message_id}: {exc}")
        return None

    headers = _headers_dict(msg.get("payload", {}))
    doc = {
        "external_id": message_id,
        "thread_id": msg.get("threadId"),
        "subject": headers.get("subject", ""),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "date": headers.get("date", ""),
        "snippet": msg.get("snippet", ""),
        "body": _extract_body(msg.get("payload", {})),
        "label_ids": msg.get("labelIds", []),
    }
    upsert("emails", {"external_id": message_id}, {**doc, "collected_at": datetime.utcnow()})
    return doc


def create_draft(to: str, subject: str, body: str, thread_id: Optional[str] = None) -> Optional[str]:
    """
    Module 7/8: creates a Gmail draft (does NOT send). Returns the Gmail
    draft id. The actual send only happens later, from
    email_assistant.wait_for_approval, once a human approves it.
    """
    service = get_gmail_service()
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    body_payload = {"message": {"raw": raw}}
    if thread_id:
        body_payload["message"]["threadId"] = thread_id

    try:
        draft = service.users().drafts().create(userId="me", body=body_payload).execute()
        return draft.get("id")
    except HttpError as exc:
        logger.error(f"Gmail create_draft failed: {exc}")
        return None


def send_draft(draft_id: str) -> bool:
    """Module 7: sends a previously-created Gmail draft. Only ever called after human approval."""
    service = get_gmail_service()
    try:
        service.users().drafts().send(userId="me", body={"id": draft_id}).execute()
        return True
    except HttpError as exc:
        logger.error(f"Gmail send_draft failed for draft_id={draft_id}: {exc}")
        return False
