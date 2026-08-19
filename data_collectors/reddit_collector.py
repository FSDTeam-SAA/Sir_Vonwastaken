"""
data_collectors/reddit_collector.py

Module 1 (Reddit part) — collects posts from a predefined list of
subreddits (config: REDDIT_SUBREDDITS) via PRAW (Reddit's official API
wrapper). Growth-relevant signals (upvotes, comments, upvote ratio, post
age) are captured so the ranking engine can compute discussion velocity.

Note: automatic discovery of *additional* relevant subreddits (mentioned
in the proposal as a "later" capability) is not implemented here — it
would depend on the similarity/embedding pipeline (Module 3) being run
first against a corpus of subreddit descriptions, which is out of scope
for this pass. `search_subreddits()` below gives a manual/AI-assisted
way to discover candidates for the creator to add to REDDIT_SUBREDDITS.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

import praw
from prawcore.exceptions import PrawcoreException

from config.settings import settings
from database.mongodb import upsert
from utils.logger import logger

_reddit_client = None


def get_reddit_client() -> praw.Reddit:
    global _reddit_client
    if _reddit_client is None:
        if not (settings.reddit_client_id and settings.reddit_client_secret):
            raise RuntimeError("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are not set. Add them to your .env file.")
        _reddit_client = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )
    return _reddit_client


def _post_to_doc(submission) -> Dict:
    return {
        "external_id": submission.id,
        "subreddit": str(submission.subreddit),
        "title": submission.title,
        "selftext": submission.selftext or "",
        "url": submission.url,
        "permalink": f"https://reddit.com{submission.permalink}",
        "author": str(submission.author) if submission.author else "[deleted]",
        "score": submission.score,
        "upvote_ratio": submission.upvote_ratio,
        "num_comments": submission.num_comments,
        "created_utc": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat(),
        "is_video": bool(submission.is_video),
        "flair": submission.link_flair_text,
    }


def _store_raw(doc: Dict) -> None:
    upsert("raw_content", {"platform": "reddit", "external_id": doc["external_id"]}, {**doc, "platform": "reddit", "collected_at": datetime.utcnow()})


def get_subreddit_posts(subreddit_name: str, limit: int = 25, listing: str = "hot") -> List[Dict]:
    """
    Module 1: pulls posts from a single subreddit. `listing` is one of
    "hot", "new", "top", "rising" — PRAW's standard listing generators.
    """
    reddit = get_reddit_client()
    try:
        subreddit = reddit.subreddit(subreddit_name)
        listing_fn = getattr(subreddit, listing)
        submissions = list(listing_fn(limit=limit))
    except PrawcoreException as exc:
        logger.error(f"Reddit fetch failed for r/{subreddit_name}: {exc}")
        return []

    docs = [_post_to_doc(s) for s in submissions]
    for doc in docs:
        _store_raw(doc)
    logger.info(f"Collected {len(docs)} posts from r/{subreddit_name} ({listing}).")
    return docs


def search_subreddits(query: str, limit: int = 10) -> List[Dict]:
    """Discovery helper: find candidate subreddits matching a keyword, for the creator to review/add."""
    reddit = get_reddit_client()
    try:
        results = reddit.subreddits.search(query, limit=limit)
        return [
            {
                "name": sr.display_name,
                "subscribers": sr.subscribers,
                "public_description": sr.public_description,
            }
            for sr in results
        ]
    except PrawcoreException as exc:
        logger.error(f"Reddit subreddit search failed for '{query}': {exc}")
        return []


def run_full_collection(limit_per_subreddit: int = 25) -> Dict[str, int]:
    """Module 1: collects from every subreddit configured in REDDIT_SUBREDDITS."""
    counts: Dict[str, int] = {}
    if not settings.reddit_subreddits:
        logger.warning("REDDIT_SUBREDDITS is empty — nothing to collect. Set it in your .env file.")
        return counts

    for subreddit_name in settings.reddit_subreddits:
        docs = get_subreddit_posts(subreddit_name, limit=limit_per_subreddit, listing="hot")
        counts[subreddit_name] = len(docs)

    logger.info(f"Reddit collection complete: {counts}")
    return counts
