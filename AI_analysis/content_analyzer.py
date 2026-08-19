"""
AI_analysis/content_analyzer.py

Module: AI Intelligence Layer. For a piece of collected content (video
title+description, Reddit post, trend keyword) this determines:
  - topic / category
  - content format (short-form video, long-form, discussion, list, etc.)
  - a handful of topic tags
  - a one-line summary

This is the "understand the topic of each trend" + "categorize content"
+ "detect content format" piece of the proposal's AI Intelligence Layer.
Similarity scoring against the creator profile (a separate concern) lives
in content_similarity_check/, and growth/engagement scoring lives in
trend_ranking/ — this module only classifies a single piece of content.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from database.mongodb import get_collection
from utils.llm_client import chat_complete_json
from utils.logger import logger

_SYSTEM_PROMPT = (
    "You are a content classification engine for a YouTube creator's trend-intelligence "
    "system. You analyze short pieces of text (a video title/description, a forum post, "
    "or a search keyword) and return structured metadata about it. Be concise and specific."
)


def analyze_content(text: str) -> Dict:
    """
    Returns: {"category": str, "format": str, "topics": [str], "summary": str}
    Returns a safe empty-ish dict (never raises) so batch pipelines don't
    break on one bad classification.
    """
    if not text or not text.strip():
        return {"category": "", "format": "", "topics": [], "summary": ""}

    prompt = f"""Analyze this piece of content:

\"\"\"{text[:1500]}\"\"\"

Determine:
1. category — a short label for its subject matter (e.g. "Technology", "Fitness", "Gaming", "Finance", "Comedy")
2. format — the likely content format (one of: "short-form video", "long-form video", "tutorial", "listicle", "discussion", "news/announcement", "story/vlog", "other")
3. topics — 3-5 specific topic keywords/tags
4. summary — one sentence summarizing what this content is about

Respond with a JSON object: {{"category": "...", "format": "...", "topics": ["...", "..."], "summary": "..."}}
Output ONLY the JSON, no prose."""

    result = chat_complete_json(_SYSTEM_PROMPT, prompt, max_tokens=250, temperature=0.3)
    return {
        "category": result.get("category", ""),
        "format": result.get("format", ""),
        "topics": result.get("topics", []),
        "summary": result.get("summary", ""),
    }


def analyze_and_store(processed_content_id, text: str) -> Dict:
    """Runs analyze_content and writes the result back onto the processed_content doc."""
    analysis = analyze_content(text)
    get_collection("processed_content").update_one(
        {"_id": processed_content_id},
        {"$set": {"analysis": analysis, "analyzed_at": datetime.utcnow()}},
    )
    return analysis


def analyze_pending_batch(limit: int = 100) -> int:
    """
    Module: AI Intelligence Layer batch runner. Analyzes every
    processed_content doc that doesn't have an `analysis` field yet.
    Returns the number of documents analyzed.
    """
    collection = get_collection("processed_content")
    docs = list(collection.find({"analysis": {"$exists": False}}).limit(limit))

    for doc in docs:
        analyze_and_store(doc["_id"], doc.get("text_for_ai", doc.get("title", "")))

    logger.info(f"AI_analysis: analyzed {len(docs)} pending content items.")
    return len(docs)
