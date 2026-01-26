"""API route modules."""

from facial_recognition.api.routes.faces import router as faces_router
from facial_recognition.api.routes.health import router as health_router
from facial_recognition.api.routes.recognize import router as recognize_router

__all__ = ["faces_router", "health_router", "recognize_router"]
