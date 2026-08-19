"""
notification_system/email.py

Module: Notification Layer (email alerts). Sends a plain-text email
notification via SMTP. This is separate from the Gmail API used in
data_collectors/gmail_collector.py and email_assistant/ (those read the
inbox and draft replies via OAuth2; this just sends a one-way alert via
a regular SMTP account, e.g. Gmail's SMTP relay with an app password).
"""
from __future__ import annotations

import smtplib
from email.mime.text import MIMEText

from config.settings import settings
from utils.logger import logger


def send(title: str, message: str) -> bool:
    if not (settings.smtp_host and settings.smtp_user and settings.smtp_password and settings.notify_email_to):
        logger.warning("SMTP settings or NOTIFY_EMAIL_TO not fully configured; skipping email notification.")
        return False

    msg = MIMEText(message)
    msg["Subject"] = title
    msg["From"] = settings.smtp_user
    msg["To"] = settings.notify_email_to

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as server:
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(settings.smtp_user, [settings.notify_email_to], msg.as_string())
        return True
    except (smtplib.SMTPException, OSError) as exc:
        logger.error(f"Email notification failed: {exc}")
        return False
