"""
notification_system/telegram.py

Module: Notification Layer (Telegram).

Implementation note: requirements.txt lists `python-telegram-bot`, whose
modern (v20+) API is fully async (`Application`/`Bot.send_message` are
coroutines) and is built around a long-running bot process — not a good
fit for firing a single notification from inside a synchronous FastAPI
request. Since sending a message is also just one HTTP POST, this module
calls the Telegram Bot HTTP API directly via `requests` (already a
dependency), which is simpler and synchronous. If you'd rather standardize
on the python-telegram-bot library specifically, say so and I'll wire it
in with `asyncio.run(...)` instead.
"""
from __future__ import annotations

import requests

from config.settings import settings
from utils.logger import logger


def send(message: str) -> bool:
    if not (settings.telegram_bot_token and settings.telegram_chat_id):
        logger.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; skipping Telegram notification.")
        return False

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    try:
        response = requests.post(
            url, json={"chat_id": settings.telegram_chat_id, "text": message[:4000]}, timeout=10
        )
        if response.status_code != 200:
            logger.error(f"Telegram send failed: {response.status_code} {response.text}")
            return False
        return True
    except requests.RequestException as exc:
        logger.error(f"Telegram notification failed: {exc}")
        return False
