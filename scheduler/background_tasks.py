"""
scheduler/background_tasks.py

Background task scheduler using APScheduler for continuous automation.

Runs independently scheduled jobs for:
  - Data collection (YouTube, Reddit, Google Trends, Gmail)
  - Content processing and analysis
  - Embedding generation
  - Trend ranking and notification
  - Email polling and sponsorship detection
"""
from __future__ import annotations

from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

from AI_analysis.content_analyzer import analyze_pending_batch
from config.settings import settings
from content_similarity_check.embedding_search import embed_pending_content
from data_collectors import (
    data_processor,
    gmail_collector,
    google_trends_collector,
    reddit_collector,
    youtube_collector,
)
from email_assistant.detect_sponsorship import scan_inbox_for_sponsorships
from email_assistant.summarize_email import summarize_pending_batch
from trend_ranking.ranking_engine import rank_trends
from utils.logger import logger

_scheduler = None


def get_scheduler() -> BackgroundScheduler:
    """Get or create the background scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler()
    return _scheduler


def start_scheduler() -> None:
    """Start the background scheduler and register all jobs."""
    scheduler = get_scheduler()
    
    if scheduler.running:
        logger.info("Scheduler already running")
        return
    
    # Data Collection Jobs
    scheduler.add_job(
        youtube_collector.run_full_collection,
        "interval",
        hours=2,
        id="collect_youtube",
        name="Collect YouTube videos",
        misfire_grace_time=300,
        coalesce=True,
    )
    
    scheduler.add_job(
        reddit_collector.run_full_collection,
        "interval",
        hours=2,
        id="collect_reddit",
        name="Collect Reddit posts",
        misfire_grace_time=300,
        coalesce=True,
    )
    
    scheduler.add_job(
        google_trends_collector.run_full_collection,
        "interval",
        hours=1,
        id="collect_trends",
        name="Collect Google Trends",
        misfire_grace_time=300,
        coalesce=True,
    )
    
    scheduler.add_job(
        gmail_collector.list_recent_messages,
        "interval",
        minutes=30,
        id="collect_gmail",
        name="Sync Gmail inbox",
        misfire_grace_time=300,
        coalesce=True,
    )
    
    # Data Processing Jobs
    scheduler.add_job(
        data_processor.process_all_platforms,
        "interval",
        hours=1,
        id="process_raw",
        name="Process raw content",
        misfire_grace_time=300,
        coalesce=True,
    )
    
    # AI Analysis Jobs
    scheduler.add_job(
        analyze_pending_batch,
        "interval",
        hours=2,
        id="analyze_content",
        name="Analyze content topics",
        misfire_grace_time=300,
        coalesce=True,
        kwargs={"limit": 100},
    )
    
    # Embedding Generation Job
    scheduler.add_job(
        embed_pending_content,
        "interval",
        hours=1,
        id="embed_content",
        name="Generate embeddings",
        misfire_grace_time=300,
        coalesce=True,
        kwargs={"limit": 100},
    )
    
    # Trend Ranking & Notification Job (most frequent)
    scheduler.add_job(
        rank_trends,
        "interval",
        minutes=30,
        id="rank_trends",
        name="Rank trends and notify",
        misfire_grace_time=300,
        coalesce=True,
        kwargs={"limit": 100, "channel_id": settings.youtube_channel_id or None},
    )
    
    # Email Processing Jobs
    scheduler.add_job(
        scan_inbox_for_sponsorships,
        "interval",
        minutes=30,
        id="detect_sponsorships",
        name="Detect sponsorship emails",
        misfire_grace_time=300,
        coalesce=True,
        kwargs={"limit": 50},
    )
    
    scheduler.add_job(
        summarize_pending_batch,
        "interval",
        minutes=45,
        id="summarize_emails",
        name="Summarize sponsorship emails",
        misfire_grace_time=300,
        coalesce=True,
        kwargs={"limit": 50},
    )
    
    # Scheduler info job (logs status every 1 hour)
    scheduler.add_job(
        log_scheduler_status,
        "interval",
        hours=1,
        id="scheduler_status",
        name="Log scheduler status",
    )
    
    scheduler.start()
    logger.info(f"Background scheduler started with {len(scheduler.get_jobs())} scheduled jobs")


def stop_scheduler() -> None:
    """Stop the background scheduler."""
    scheduler = get_scheduler()
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Background scheduler stopped")


def get_jobs_status() -> list:
    """Get status of all scheduled jobs."""
    scheduler = get_scheduler()
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "trigger": str(job.trigger),
            "next_run_time": str(job.next_run_time),
        })
    return jobs


def log_scheduler_status() -> None:
    """Log current scheduler status (for monitoring)."""
    scheduler = get_scheduler()
    if not scheduler.running:
        logger.warning("Scheduler is NOT running!")
        return
    
    jobs = scheduler.get_jobs()
    logger.info(f"Scheduler Status: {len(jobs)} jobs running")
    for job in jobs:
        logger.debug(f"  - {job.name}: next run at {job.next_run_time}")
