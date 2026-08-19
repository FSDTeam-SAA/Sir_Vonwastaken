"""
test_system.py

End-to-end smoke test for the Content Trend Intelligence Assistant API.
Runs through the full pipeline in the correct order (each step depends
on data produced by the one before it) and prints a clear PASS/FAIL
summary at the end, along with exactly which .env credential is likely
broken if something fails.

Usage:
    pip install requests
    python test_system.py

Edit BASE_URL and CHANNEL_ID below before running.
"""
from __future__ import annotations

import sys
import time
from typing import Any, Dict, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration — edit these two before running
# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8000"
CHANNEL_ID = "YOUR_CHANNEL_ID"          # your own YouTube channel ID (UC...)

# Set to True only if you've completed the Gmail OAuth setup (credentials.json in place)
TEST_GMAIL = False

# Which notification channels to test (must match what you enabled in .env)
NOTIFY_CHANNELS = ["desktop"]           # e.g. ["desktop", "discord", "telegram", "email"]

# ---------------------------------------------------------------------------

results = []  # (step_name, passed: bool, detail: str)


def record(step: str, passed: bool, detail: str = "") -> None:
    results.append((step, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {step}{' — ' + detail if detail else ''}")


def call(method: str, path: str, **kwargs) -> Optional[requests.Response]:
    url = f"{BASE_URL}{path}"
    try:
        return requests.request(method, url, timeout=90, **kwargs)
    except requests.RequestException as exc:
        print(f"    -> request error: {exc}")
        return None


def step(name: str, method: str, path: str, expect_key: Optional[str] = None, **kwargs) -> Optional[Dict[str, Any]]:
    print(f"\n--- {name} ---")
    resp = call(method, path, **kwargs)
    if resp is None:
        record(name, False, "no response (server unreachable?)")
        return None

    if resp.status_code >= 400:
        record(name, False, f"HTTP {resp.status_code}: {resp.text[:200]}")
        return None

    try:
        data = resp.json()
    except ValueError:
        record(name, False, "response was not valid JSON")
        return None

    if expect_key and isinstance(data, dict) and expect_key not in data:
        record(name, False, f"expected key '{expect_key}' missing from response")
        return data

    record(name, True)
    return data


def main() -> None:
    print(f"Testing {BASE_URL} ...")

    # 0. Server + MongoDB
    step("Server is running", "GET", "/")
    health = step("MongoDB connection", "GET", "/api/health", expect_key="mongodb_connected")
    if not health or not health.get("mongodb_connected"):
        print("\nMongoDB is not connected — fix MONGODB_URI in .env before continuing. Stopping.")
        print_summary()
        sys.exit(1)

    # 1. Creator profile (tests YOUTUBE_API_KEY + CHANNEL_ID)
    profile = step(
        "Build creator profile",
        "POST",
        f"/api/creator-profile/{CHANNEL_ID}/build",
    )

    # 2. Collection (tests YouTube / Reddit / Google Trends credentials)
    collected = step("Collect content from all platforms", "POST", "/api/collect/all")
    time.sleep(1)

    # 3. Processing
    step("Process/normalize collected content", "POST", "/api/process/run")

    # 4. AI classification (tests OPENAI_API_KEY)
    step("AI content classification", "POST", "/api/analysis/run")

    # 5. Embeddings + ranking (tests OPENAI_API_KEY embeddings + scoring engine)
    step("Embed pending content", "POST", "/api/similarity/embed-pending")
    trends = step(
        "Rank trends",
        "POST",
        f"/api/trends/rank?channel_id={CHANNEL_ID}",
    )

    # 6. Content generation (tests OPENAI_API_KEY chat completions)
    if trends and isinstance(trends, list) and len(trends) > 0:
        content_id = trends[0]["content_id"]
        step(
            "Generate content for top trend",
            "POST",
            f"/api/content/generate/{content_id}?channel_id={CHANNEL_ID}",
        )
    else:
        record("Generate content for top trend", False, "skipped — no ranked trends available yet")

    # 7. Notifications
    step(
        "Send test notification",
        "POST",
        "/api/notify/test",
        json={"title": "Test alert", "message": "System check", "channels": NOTIFY_CHANNELS},
    )

    # 8. Gmail / email assistant (optional — only if Gmail OAuth is set up)
    if TEST_GMAIL:
        step("Sync Gmail", "POST", "/api/collect/gmail/sync")
        step("Scan inbox for sponsorship emails", "POST", "/api/emails/scan-sponsorships")
    else:
        print("\n(Skipping Gmail tests — set TEST_GMAIL = True once credentials.json is in place.)")

    print_summary()


def print_summary() -> None:
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, detail in results:
        mark = "✅" if ok else "❌"
        print(f"{mark} {name}" + (f" — {detail}" if detail else ""))
    print("-" * 50)
    print(f"{passed}/{total} steps passed")
    if passed < total:
        print("\nSome steps failed — check the .env credential related to that step (see README.md's Environment Variables table).")


if __name__ == "__main__":
    main()