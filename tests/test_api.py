"""Tests for REST API endpoints."""

import pytest
from httpx import AsyncClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client: AsyncClient) -> None:
        """Test root endpoint returns health info."""
        response = await async_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client: AsyncClient) -> None:
        """Test health endpoint."""
        response = await async_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "persons_registered" in data


class TestFaceEndpoints:
    """Tests for face management endpoints."""

    @pytest.mark.asyncio
    async def test_list_persons_empty(self, async_client: AsyncClient) -> None:
        """Test listing persons when empty."""
        response = await async_client.get("/faces")

        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_get_person_not_found(self, async_client: AsyncClient) -> None:
        """Test getting non-existent person."""
        response = await async_client.get(
            "/faces/00000000-0000-0000-0000-000000000000"
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_person_not_found(self, async_client: AsyncClient) -> None:
        """Test deleting non-existent person."""
        response = await async_client.delete(
            "/faces/00000000-0000-0000-0000-000000000000"
        )

        assert response.status_code == 404


class TestRecognitionEndpoints:
    """Tests for recognition endpoints."""

    @pytest.mark.asyncio
    async def test_detect_no_file(self, async_client: AsyncClient) -> None:
        """Test detect endpoint without file."""
        response = await async_client.post("/recognize/detect")

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_recognize_no_file(self, async_client: AsyncClient) -> None:
        """Test recognize endpoint without file."""
        response = await async_client.post("/recognize")

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_verify_not_found(self, async_client: AsyncClient) -> None:
        """Test verify endpoint with non-existent person."""
        response = await async_client.post(
            "/recognize/verify/00000000-0000-0000-0000-000000000000",
            files={"image": ("test.jpg", b"fake image data", "image/jpeg")},
        )

        assert response.status_code in (400, 404)
