"""Health check endpoints."""

from fastapi import APIRouter, Depends

from facial_recognition import __version__
from facial_recognition.api.dependencies import get_recognizer
from facial_recognition.api.schemas import HealthResponse
from facial_recognition.core.recognizer import FaceRecognizer

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is healthy and get basic stats.",
)
async def health_check(
    recognizer: FaceRecognizer = Depends(get_recognizer),
) -> HealthResponse:
    """Return service health status."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        persons_registered=recognizer.person_count,
    )


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Root endpoint",
    description="Root endpoint with service info.",
)
async def root(
    recognizer: FaceRecognizer = Depends(get_recognizer),
) -> HealthResponse:
    """Return service info at root."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        persons_registered=recognizer.person_count,
    )
