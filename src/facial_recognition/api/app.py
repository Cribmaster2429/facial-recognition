"""FastAPI application factory and configuration."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from facial_recognition import __version__
from facial_recognition.api.dependencies import initialize_app, shutdown_app
from facial_recognition.api.routes import faces_router, health_router, recognize_router
from facial_recognition.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown events."""
    await initialize_app()
    yield
    await shutdown_app()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Facial Recognition API",
        description=(
            "Real-time facial recognition system using deep learning. "
            "Provides endpoints for face detection, registration, and recognition."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(faces_router)
    app.include_router(recognize_router)

    return app


app = create_app()
