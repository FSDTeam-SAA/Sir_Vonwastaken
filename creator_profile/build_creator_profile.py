"""
creator_profile/build_creator_profile.py

Module 2 — Creator Style Learning.

Builds a "Creator Profile" from a creator's historical YouTube content
(titles, descriptions, categories/topics derived via AI_analysis) and
stores:
  1. A single averaged embedding vector for the creator's overall style
     (used by content_similarity_check for fast similarity scoring).
  2. A summary document (top categories/topics, video count, avg
     engagement) in MongoDB `creator_profiles`, used to give the AI
     generator style context (see AI_generator/generate_content.py).

This replaces the original template's `generate_embedding` /
`store_embedding` placeholders with the shared backend selection in
content_similarity_check.embedding_search and persistence in
database.vector_store.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from AI_analysis.content_analyzer import analyze_content
from content_similarity_check.embedding_search import generate_embeddings
from data_collectors.youtube_collector import get_channel_videos
from database.mongodb import find_one, strip_id, upsert
from database.vector_store import get_embedding as get_stored_embedding
from database.vector_store import get_embedding_record as get_stored_embedding_record
from database.vector_store import store_embedding
from utils.logger import logger

_NAMESPACE = "creator_profile"


def _store_profile_embedding(
    creator_id: str,
    vector: List[float],
    embedding_space: str,
    video_count: int,
) -> None:
    """Store a profile vector with enough metadata to reject incompatible content vectors."""
    store_embedding(
        _NAMESPACE,
        creator_id,
        vector,
        metadata={
            "video_count": video_count,
            "embedding_space": embedding_space,
            "embedding_dimension": len(vector),
        },
    )


def build_creator_profile(creator_data: Dict) -> List[float]:
    """
    Builds a creator profile from an explicit data dict (matches the
    original template's contract), for cases where the caller already
    has the creator's titles/descriptions/transcripts/categories/topics
    on hand (e.g. bulk import) rather than pulling live from YouTube.

    creator_data = {
        "creator_id": str,
        "titles": [str], "descriptions": [str], "transcripts": [str],
        "categories": [str], "topics": [str],
    }
    """
    creator_id = creator_data["creator_id"]
    titles = creator_data.get("titles", [])
    descriptions = creator_data.get("descriptions", [])
    transcripts = creator_data.get("transcripts", [])
    categories = creator_data.get("categories", [])
    topics = creator_data.get("topics", [])

    combined_text = " ".join(titles + descriptions + transcripts + categories + topics).strip()
    if not combined_text:
        raise ValueError("build_creator_profile: creator_data produced no text to embed.")

    vectors, embedding_space = generate_embeddings([combined_text])
    if not vectors or not embedding_space:
        raise RuntimeError(
            f"Could not generate a creator-profile embedding for creator_id={creator_id}."
        )
    creator_profile_embedding = vectors[0]
    _store_profile_embedding(creator_id, creator_profile_embedding, embedding_space, len(titles))

    upsert(
        "creator_profiles",
        {"channel_id": creator_id},
        {
            "channel_id": creator_id,
            "top_categories": [c for c, _ in Counter(categories).most_common(5)],
            "top_topics": [t for t, _ in Counter(topics).most_common(10)],
            "video_count": len(titles),
            "updated_at": datetime.utcnow(),
        },
    )
    logger.info(f"Built creator profile for creator_id={creator_id} from {len(titles)} videos.")
    return creator_profile_embedding


def build_creator_profile_from_channel(channel_id: str, max_videos: int = 25) -> Optional[List[float]]:
    """
    Module 2, primary path: pulls the creator's own recent uploads via
    the YouTube collector, runs each through AI_analysis to derive
    category/topic/format, and builds the profile from that.
    """
    videos = get_channel_videos(channel_id, max_results=max_videos)
    if not videos:
        logger.warning(f"No videos found for channel_id={channel_id}; cannot build profile.")
        return None

    titles = [v["title"] for v in videos]
    descriptions = [v["description"][:500] for v in videos]

    categories: List[str] = []
    topics: List[str] = []
    engagement_scores: List[float] = []

    for video in videos:
        analysis = analyze_content(f"{video['title']}\n{video['description'][:500]}")
        if analysis.get("category"):
            categories.append(analysis["category"])
        topics.extend(analysis.get("topics", []))
        views = video.get("view_count", 0) or 0
        likes = video.get("like_count", 0) or 0
        engagement_scores.append(likes / views if views else 0.0)

    combined_texts = titles + descriptions
    vectors, embedding_space = generate_embeddings(combined_texts)
    if not vectors or not embedding_space:
        logger.warning(f"Embedding batch returned nothing for channel_id={channel_id}.")
        return None

    profile_embedding = np.mean(np.array(vectors), axis=0).tolist()
    _store_profile_embedding(channel_id, profile_embedding, embedding_space, len(videos))

    upsert(
        "creator_profiles",
        {"channel_id": channel_id},
        {
            "channel_id": channel_id,
            "top_categories": [c for c, _ in Counter(categories).most_common(5)],
            "top_topics": [t for t, _ in Counter(topics).most_common(10)],
            "video_count": len(videos),
            "avg_engagement_rate": float(np.mean(engagement_scores)) if engagement_scores else 0.0,
            "updated_at": datetime.utcnow(),
        },
    )
    logger.info(f"Built creator profile for channel_id={channel_id} from {len(videos)} live YouTube videos.")
    return profile_embedding


def get_creator_profile_embedding(channel_id: str) -> Optional[List[float]]:
    return get_stored_embedding(_NAMESPACE, channel_id)


def get_creator_profile_embedding_record(channel_id: str) -> Optional[Tuple[List[float], Dict]]:
    """Retrieve a creator vector and its compatibility metadata atomically."""
    return get_stored_embedding_record(_NAMESPACE, channel_id)


def get_creator_profile_summary(channel_id: str) -> Optional[Dict]:
    """Used by AI_generator.generate_content to give the LLM style context."""
    return strip_id(find_one("creator_profiles", {"channel_id": channel_id}))
