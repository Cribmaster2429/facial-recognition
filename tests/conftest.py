"""Pytest fixtures for facial recognition tests."""

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator
from uuid import uuid4

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from facial_recognition.api.app import create_app
from facial_recognition.core.models import FaceEncoding, Person
from facial_recognition.storage.memory import MemoryStorage


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_encoding() -> FaceEncoding:
    """Create a sample 128-dimensional face encoding."""
    np.random.seed(42)
    vector = np.random.randn(128).astype(np.float64)
    vector = vector / np.linalg.norm(vector)
    return FaceEncoding(vector=vector)


@pytest.fixture
def sample_encoding_similar(sample_encoding: FaceEncoding) -> FaceEncoding:
    """Create an encoding similar to sample_encoding."""
    noise = np.random.randn(128) * 0.1
    vector = sample_encoding.vector + noise
    vector = vector / np.linalg.norm(vector)
    return FaceEncoding(vector=vector)


@pytest.fixture
def sample_encoding_different() -> FaceEncoding:
    """Create a very different encoding."""
    np.random.seed(123)
    vector = np.random.randn(128).astype(np.float64)
    vector = vector / np.linalg.norm(vector)
    return FaceEncoding(vector=vector)


@pytest.fixture
def sample_person(sample_encoding: FaceEncoding) -> Person:
    """Create a sample person."""
    return Person(
        id=uuid4(),
        name="John Doe",
        encoding=sample_encoding,
        metadata={"department": "Engineering"},
    )


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a simple test image (no actual face)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
async def memory_storage() -> AsyncGenerator[MemoryStorage, None]:
    """Create an in-memory storage for testing."""
    storage = MemoryStorage()
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for API testing."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_faces.db"
