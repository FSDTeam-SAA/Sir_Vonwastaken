"""
notification_system/desktop_notifications.py

Module: Notification Layer (native macOS desktop notifications).
Matches the proposal's deployment model — the whole system runs locally
on the creator's Mac — by shelling out to `osascript`, which is what
actually produces a native Notification Center banner on macOS. No
extra Python package is required for this.
"""
from __future__ import annotations

import platform
import subprocess

from utils.logger import logger


def send(title: str, message: str) -> bool:
    if platform.system() != "Darwin":
        logger.warning(f"Desktop notifications only supported on macOS (current OS: {platform.system()}). Skipped: {title}")
        return False

    # Escape double quotes so a trend title containing one doesn't break the AppleScript string.
    safe_title = title.replace('"', '\\"')
    safe_message = message.replace('"', '\\"')
    script = f'display notification "{safe_message}" with title "{safe_title}"'

    try:
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.error(f"Desktop notification failed: {exc}")
        return False
