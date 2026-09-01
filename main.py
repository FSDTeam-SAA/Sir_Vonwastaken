"""
main.py

FastAPI application entrypoint.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Interactive API docs are then available at http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from config.settings import settings
from database.mongodb import ensure_indexes
from scheduler.background_tasks import get_scheduler, start_scheduler, stop_scheduler
from utils.logger import logger

app = FastAPI(
    title=settings.app_name,
    description="AI-powered content trend intelligence & brand management assistant.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.on_event("startup")
def on_startup() -> None:
    logger.info(f"Starting {settings.app_name} (env={settings.app_env})")
    try:
        ensure_indexes()
    except Exception as exc:  # noqa: BLE001 — don't crash startup if Mongo isn't reachable yet
        logger.error(f"Could not ensure MongoDB indexes on startup: {exc}")
    
    # Start background scheduler if enabled
    if settings.scheduler_enabled:
        try:
            start_scheduler()
            logger.info("Background scheduler started successfully")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to start background scheduler: {exc}")


@app.on_event("shutdown")
def on_shutdown() -> None:
    logger.info(f"Shutting down {settings.app_name}")
    # Stop background scheduler
    try:
        stop_scheduler()
        logger.info("Background scheduler stopped")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Error stopping scheduler: {exc}")


@app.get("/")
def root():
    scheduler = get_scheduler()
    return {
        "app": settings.app_name,
        "status": "running",
        "docs": "/docs",
        "api_prefix": "/api",
        "scheduler_status": "running" if scheduler.running else "stopped",
    }
