"""
config/settings.py

Central configuration module. Loads all runtime configuration from
environment variables (via a `.env` file at the project root) so no
secrets are ever hard-coded in source.

Every other module in the codebase imports `settings` from here rather
than calling `os.getenv` directly, so there is a single source of truth
for configuration and a single place to see everything the app needs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

# Load .env from project root (no-op if it doesn't exist / vars already set)
load_dotenv()


def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_list(name: str, default: str = "") -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


@dataclass
class Settings:
    # ---------------------------------------------------------------
    # App
    # ---------------------------------------------------------------
    app_name: str = os.getenv("APP_NAME", "Content Trend Intelligence Assistant")
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # ---------------------------------------------------------------
    # MongoDB
    # ---------------------------------------------------------------
    mongodb_uri: str = os.getenv("MONGODB_URI", "")
    mongodb_db_name: str = os.getenv("MONGODB_DB_NAME", "content_trend_assistant")

    # ---------------------------------------------------------------
    # OpenAI (chat + embeddings)
    # ---------------------------------------------------------------
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # ---------------------------------------------------------------
    # YouTube Data API
    # ---------------------------------------------------------------
    youtube_api_key: str = os.getenv("YOUTUBE_API_KEY", "")
    youtube_channel_id: str = os.getenv("YOUTUBE_CHANNEL_ID", "")
    youtube_watch_channels: List[str] = field(default_factory=lambda: _get_list("YOUTUBE_WATCH_CHANNELS"))
    youtube_watch_queries: List[str] = field(default_factory=lambda: _get_list("YOUTUBE_WATCH_QUERIES"))

    # ---------------------------------------------------------------
    # Reddit (PRAW)
    # ---------------------------------------------------------------
    reddit_client_id: str = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_client_secret: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    reddit_user_agent: str = os.getenv("REDDIT_USER_AGENT", "content-trend-assistant/1.0")
    reddit_subreddits: List[str] = field(default_factory=lambda: _get_list("REDDIT_SUBREDDITS"))

    # ---------------------------------------------------------------
    # Google Trends (pytrends — no API key required)
    # ---------------------------------------------------------------
    google_trends_geo: str = os.getenv("GOOGLE_TRENDS_GEO", "US")
    google_trends_keywords: List[str] = field(default_factory=lambda: _get_list("GOOGLE_TRENDS_KEYWORDS"))

    # ---------------------------------------------------------------
    # Gmail API (OAuth2 — installed-app flow)
    # ---------------------------------------------------------------
    gmail_credentials_file: str = os.getenv("GMAIL_CREDENTIALS_FILE", "credentials.json")
    gmail_token_file: str = os.getenv("GMAIL_TOKEN_FILE", "token.json")
    gmail_query: str = os.getenv("GMAIL_QUERY", "newer_than:2d")

    # ---------------------------------------------------------------
    # Notifications
    # ---------------------------------------------------------------
    notify_desktop_enabled: bool = field(default_factory=lambda: _get_bool("NOTIFY_DESKTOP_ENABLED", True))
    notify_discord_enabled: bool = field(default_factory=lambda: _get_bool("NOTIFY_DISCORD_ENABLED", False))
    notify_telegram_enabled: bool = field(default_factory=lambda: _get_bool("NOTIFY_TELEGRAM_ENABLED", False))
    notify_email_enabled: bool = field(default_factory=lambda: _get_bool("NOTIFY_EMAIL_ENABLED", False))

    discord_webhook_url: str = os.getenv("DISCORD_WEBHOOK_URL", "")

    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = field(default_factory=lambda: _get_int("SMTP_PORT", 587))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    notify_email_to: str = os.getenv("NOTIFY_EMAIL_TO", "")

    # ---------------------------------------------------------------
    # Trend scoring thresholds / weights
    # ---------------------------------------------------------------
    trend_notify_score_threshold: float = field(default_factory=lambda: _get_float("TREND_NOTIFY_SCORE_THRESHOLD", 0.7))
    trend_weight_growth: float = field(default_factory=lambda: _get_float("TREND_WEIGHT_GROWTH", 0.25))
    trend_weight_engagement: float = field(default_factory=lambda: _get_float("TREND_WEIGHT_ENGAGEMENT", 0.2))
    trend_weight_freshness: float = field(default_factory=lambda: _get_float("TREND_WEIGHT_FRESHNESS", 0.15))
    trend_weight_similarity: float = field(default_factory=lambda: _get_float("TREND_WEIGHT_SIMILARITY", 0.3))
    trend_weight_cross_platform: float = field(default_factory=lambda: _get_float("TREND_WEIGHT_CROSS_PLATFORM", 0.1))


settings = Settings()
