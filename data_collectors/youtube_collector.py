"""
data_collectors/youtube_collector.py

Module 1 (YouTube part) — collects trending videos, search results, and
channel uploads via the official YouTube Data API v3, plus video-level
statistics/metadata. Raw results are stored into MongoDB `raw_content`
(platform="youtube") for the data_processor to clean up downstream.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config.settings import settings
from database.mongodb import get_collection, upsert
from utils.logger import logger

_youtube_client = None


def get_youtube_client():
    global _youtube_client
    if _youtube_client is None:
        if not settings.youtube_api_key:
            raise RuntimeError("YOUTUBE_API_KEY is not set. Add it to your .env file.")
        _youtube_client = build("youtube", "v3", developerKey=settings.youtube_api_key)
    return _youtube_client


def _store_raw(video: Dict) -> None:
    upsert(
        "raw_content",
        {"platform": "youtube", "external_id": video["external_id"]},
        {**video, "platform": "youtube", "collected_at": datetime.utcnow()},
    )


def _video_to_doc(item: Dict, source: str) -> Dict:
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    content_details = item.get("contentDetails", {})
    return {
        "external_id": item["id"],
        "source": source,  # "trending" | "search" | "channel"
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "channel_id": snippet.get("channelId", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "published_at": snippet.get("publishedAt", ""),
        "category_id": snippet.get("categoryId", ""),
        "tags": snippet.get("tags", []),
        "thumbnail_url": (snippet.get("thumbnails", {}).get("high") or {}).get("url", ""),
        "duration": content_details.get("duration", ""),
        "view_count": int(stats.get("viewCount", 0) or 0),
        "like_count": int(stats.get("likeCount", 0) or 0),
        "comment_count": int(stats.get("commentCount", 0) or 0),
    }


def get_trending_videos(region_code: str = "US", category_id: Optional[str] = None, max_results: int = 25) -> List[Dict]:
    """Module 1: YouTube trending videos (used as a proxy for Shorts/trending feed)."""
    yt = get_youtube_client()
    params = dict(
        part="snippet,statistics,contentDetails",
        chart="mostPopular",
        regionCode=region_code,
        maxResults=min(max_results, 50),
    )
    if category_id:
        params["videoCategoryId"] = category_id
    try:
        response = yt.videos().list(**params).execute()
    except HttpError as exc:
        logger.error(f"YouTube trending fetch failed: {exc}")
        return []

    docs = [_video_to_doc(item, "trending") for item in response.get("items", [])]
    for doc in docs:
        _store_raw(doc)
    logger.info(f"Collected {len(docs)} trending YouTube videos.")
    return docs


def search_videos(query: str, max_results: int = 25, order: str = "relevance") -> List[Dict]:
    """Module 1: search-based discovery for a watched keyword/query."""
    yt = get_youtube_client()
    try:
        search_resp = yt.search().list(
            part="id", q=query, type="video", maxResults=min(max_results, 50), order=order
        ).execute()
        video_ids = [item["id"]["videoId"] for item in search_resp.get("items", []) if item.get("id", {}).get("videoId")]
        if not video_ids:
            return []
        details_resp = yt.videos().list(part="snippet,statistics,contentDetails", id=",".join(video_ids)).execute()
    except HttpError as exc:
        logger.error(f"YouTube search fetch failed for query='{query}': {exc}")
        return []

    docs = [_video_to_doc(item, "search") for item in details_resp.get("items", [])]
    for doc in docs:
        doc["matched_query"] = query
        _store_raw(doc)
    logger.info(f"Collected {len(docs)} YouTube videos for query='{query}'.")
    return docs


def get_channel_videos(channel_id: str, max_results: int = 25) -> List[Dict]:
    """Module 2 support: pulls a channel's recent uploads (used to build the creator profile)."""
    yt = get_youtube_client()
    try:
        channel_resp = yt.channels().list(part="contentDetails", id=channel_id).execute()
        items = channel_resp.get("items", [])
        if not items:
            logger.warning(f"No YouTube channel found for channel_id={channel_id}")
            return []
        uploads_playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

        playlist_resp = yt.playlistItems().list(
            part="contentDetails", playlistId=uploads_playlist_id, maxResults=min(max_results, 50)
        ).execute()
        video_ids = [pi["contentDetails"]["videoId"] for pi in playlist_resp.get("items", [])]
        if not video_ids:
            return []
        details_resp = yt.videos().list(part="snippet,statistics,contentDetails", id=",".join(video_ids)).execute()
    except HttpError as exc:
        logger.error(f"YouTube channel fetch failed for channel_id={channel_id}: {exc}")
        return []

    docs = [_video_to_doc(item, "channel") for item in details_resp.get("items", [])]
    for doc in docs:
        _store_raw(doc)
    logger.info(f"Collected {len(docs)} uploads for channel_id={channel_id}.")
    return docs


def get_video_transcript(video_id: str) -> Optional[str]:
    """
    Best-effort transcript fetch. The YouTube Data API v3 does not expose
    caption text directly (only caption track metadata, and downloading
    track bodies requires the uploading channel's OAuth consent), so this
    returns None when no transcript is available rather than guessing.
    """
    try:
        yt = get_youtube_client()
        captions_resp = yt.captions().list(part="snippet", videoId=video_id).execute()
        if not captions_resp.get("items"):
            return None
        logger.info(f"Caption track exists for video {video_id}, but downloading requires channel-owner OAuth — skipping.")
        return None
    except HttpError as exc:
        logger.warning(f"Could not list captions for video {video_id}: {exc}")
        return None


def run_full_collection() -> Dict[str, int]:
    """
    Orchestrates Module 1 for YouTube: trending + all configured watch
    queries + (optionally) the creator's own channel uploads.
    """
    counts = {"trending": 0, "search": 0, "channel": 0}
    counts["trending"] = len(get_trending_videos())

    for query in settings.youtube_watch_queries:
        counts["search"] += len(search_videos(query))

    for channel_id in settings.youtube_watch_channels:
        counts["channel"] += len(get_channel_videos(channel_id))

    logger.info(f"YouTube collection complete: {counts}")
    return counts
