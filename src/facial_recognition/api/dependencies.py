"""FastAPI dependencies for dependency injection."""

from functools import lru_cache
from typing import AsyncGenerator

from facial_recognition.core.detector import FaceDetector
from facial_recognition.core.encoder import FaceEncoder
from facial_recognition.core.recognizer import FaceRecognizer
from facial_recognition.storage.base import BaseStorage
from facial_recognition.storage.sqlite import SQLiteStorage


_storage: BaseStorage | None = None
_recognizer: FaceRecognizer | None = None


@lru_cache
def get_detector() -> FaceDetector:
    """Get cached face detector instance."""
    return FaceDetector()


@lru_cache
def get_encoder() -> FaceEncoder:
    """Get cached face encoder instance."""
    return FaceEncoder()


def get_recognizer() -> FaceRecognizer:
    """Get the face recognizer instance."""
    global _recognizer
    if _recognizer is None:
        _recognizer = FaceRecognizer(
            detector=get_detector(),
            encoder=get_encoder(),
        )
    return _recognizer


async def get_storage() -> AsyncGenerator[BaseStorage, None]:
    """Get storage instance as async dependency."""
    global _storage
    if _storage is None:
        _storage = SQLiteStorage()
        await _storage.initialize()
    yield _storage


async def initialize_app() -> None:
    """Initialize application resources on startup."""
    global _storage, _recognizer

    _storage = SQLiteStorage()
    await _storage.initialize()

    _recognizer = FaceRecognizer(
        detector=get_detector(),
        encoder=get_encoder(),
    )

    persons = await _storage.get_all_persons()
    _recognizer.load_persons(persons)


async def shutdown_app() -> None:
    """Clean up application resources on shutdown."""
    global _storage
    if _storage:
        await _storage.close()
        _storage = None
