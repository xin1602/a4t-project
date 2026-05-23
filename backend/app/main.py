"""
FastAPI application entry point for A4T Dashboard backend.

Startup validation:
  - Calls ModelService.get_instance() during lifespan startup.
  - If any required pkl file is missing, logs a critical error and re-raises
    the FileNotFoundError so uvicorn refuses to start.

Usage:
  uvicorn backend.app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
load_dotenv()  # load .env from project root before anything else

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers import cases, explain, graph, llm, overview, predict, users
from backend.app.services.model_service import ModelService

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.

    Startup:
      - Loads all required pkl models via ModelService.get_instance().
      - If any pkl file is missing, logs a critical message and re-raises
        FileNotFoundError to abort startup.

    Shutdown:
      - No explicit cleanup required for MVP.
    """
    # --- startup ---
    logger.info("Starting A4T Dashboard backend — loading models...")
    try:
        ModelService.get_instance()
        logger.info("All models loaded successfully. Server is ready.")
    except FileNotFoundError as exc:
        logger.critical("Startup failed: missing pkl files: %s", exc)
        raise  # re-raise so uvicorn exits with a non-zero code

    yield  # application runs here

    # --- shutdown ---
    logger.info("A4T Dashboard backend shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="A4T Fraud Detection Dashboard API",
    version="1.0.0",
    description="Backend API for the A4T fraud detection dashboard (MVP).",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow Vite dev server (http://localhost:5173)
# ---------------------------------------------------------------------------

_allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

# In production, CloudFront handles CORS; allow all origins as fallback
import os
if os.environ.get("AWS_EXECUTION_ENV") or os.environ.get("S3_BUCKET"):
    _allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(overview.router)
app.include_router(users.router)
app.include_router(graph.router)
app.include_router(cases.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(llm.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
def health_check() -> dict:
    """Return a simple liveness probe response."""
    return {"status": "ok"}
