"""
notification_system/discord.py

Module: Notification Layer (Discord). Sends a message to a Discord
channel via an incoming webhook (DISCORD_WEBHOOK_URL) using the
`discord-webhook` package already listed in requirements.txt.
"""
from __future__ import annotations

from discord_webhook import DiscordWebhook

from config.settings import settings
from utils.logger import logger


def send(title: str, message: str) -> bool:
    if not settings.discord_webhook_url:
        logger.warning("DISCORD_WEBHOOK_URL is not set; skipping Discord notification.")
        return False

    content = f"**{title}**\n{message}"
    try:
        webhook = DiscordWebhook(url=settings.discord_webhook_url, content=content[:2000])
        response = webhook.execute()
        ok = getattr(response, "status_code", 0) < 300
        if not ok:
            logger.error(f"Discord webhook returned status {getattr(response, 'status_code', 'unknown')}")
        return ok
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Discord notification failed: {exc}")
        return False
