"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class BoundingBoxResponse(BaseModel):
    """Bounding box coordinates in API response."""

    top: int
    right: int
    bottom: int
    left: int
    width: int
    height: int


class PersonCreate(BaseModel):
    """Request schema for creating a person."""

    name: str = Field(..., min_length=1, max_length=255, description="Person's name")
    metadata: dict[str, str] = Field(default_factory=dict, description="Optional metadata")


class PersonResponse(BaseModel):
    """Response schema for a person."""

    id: UUID
    name: str
    metadata: dict[str, str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PersonUpdate(BaseModel):
    """Request schema for updating a person."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    metadata: Optional[dict[str, str]] = None


class DetectedFaceResponse(BaseModel):
    """Response schema for a detected face."""

    bounding_box: BoundingBoxResponse
    confidence: float


class MatchResponse(BaseModel):
    """Response schema for a face match."""

    person: PersonResponse
    distance: float
    confidence: float


class RecognitionResultResponse(BaseModel):
    """Response schema for a recognition result."""

    face: DetectedFaceResponse
    match: Optional[MatchResponse] = None
    recognized: bool


class RecognizeResponse(BaseModel):
    """Response schema for recognition endpoint."""

    faces_detected: int
    faces_recognized: int
    results: list[RecognitionResultResponse]


class VerifyRequest(BaseModel):
    """Request schema for verification."""

    person_id: UUID = Field(..., description="ID of person to verify against")


class VerifyResponse(BaseModel):
    """Response schema for verification."""

    verified: bool
    confidence: float
    person_id: UUID


class DetectResponse(BaseModel):
    """Response schema for detection endpoint."""

    faces_detected: int
    faces: list[DetectedFaceResponse]


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str
    version: str
    persons_registered: int


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    detail: str
