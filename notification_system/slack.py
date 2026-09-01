"""
notification_system/slack.py

Module: Notification Layer (Slack). Sends messages to a Slack channel
via the Slack SDK using a webhook URL or bot token.
"""
from __future__ import annotations

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from config.settings import settings
from utils.logger import logger


def send(title: str, message: str) -> bool:
    """
    Send a notification to Slack.
    
    Args:
        title: Message title/header
        message: Message body
    
    Returns:
        True if sent successfully, False otherwise
    """
    if not settings.slack_webhook_url and not settings.slack_bot_token:
        logger.warning("SLACK_WEBHOOK_URL or SLACK_BOT_TOKEN not set; skipping Slack notification.")
        return False
    
    try:
        # Format message
        content = f"*{title}*\n{message}"
        
        # Use webhook if available (simpler)
        if settings.slack_webhook_url:
            import urllib.request
            import json
            
            payload = json.dumps({
                "text": content,
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": content}
                    }
                ]
            })
            
            req = urllib.request.Request(
                settings.slack_webhook_url,
                data=payload.encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        
        # Use bot token if webhook not available
        elif settings.slack_bot_token:
            client = WebClient(token=settings.slack_bot_token)
            
            # Get channel from config or default
            channel = getattr(settings, 'slack_channel', '#alerts')
            
            response = client.chat_postMessage(
                channel=channel,
                text=content,
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": content}
                    }
                ]
            )
            return response["ok"]
    
    except SlackApiError as exc:
        logger.error(f"Slack API error: {exc.response['error']}")
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Slack notification failed: {exc}")
        return False
