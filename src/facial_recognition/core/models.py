"""Data models for facial recognition system."""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Current time as a timezone aware UTC datetime."""
    return datetime.now(timezone.utc)


class BoundingBox(BaseModel):
    """Bounding box coordinates for a detected face."""

    top: int
    right: int
    bottom: int
    left: int

    @property
    def width(self) -> int:
        """Calculate bounding box width."""
        return self.right - self.left

    @property
    def height(self) -> int:
        """Calculate bounding box height."""
        return self.bottom - self.top

    @property
    def area(self) -> int:
        """Calculate bounding box area."""
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        """Calculate bounding box center point."""
        return (self.left + self.width // 2, self.top + self.height // 2)

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Convert to (top, right, bottom, left) tuple."""
        return (self.top, self.right, self.bottom, self.left)

    def to_cv2_rect(self) -> tuple[int, int, int, int]:
        """Convert to OpenCV rectangle format (x, y, w, h)."""
        return (self.left, self.top, self.width, self.height)


class FaceEncoding(BaseModel):
    """128-dimensional face encoding vector."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector: np.ndarray = Field(..., description="128-dimensional encoding vector")

    @classmethod
    def from_list(cls, values: list[float]) -> "FaceEncoding":
        """Create encoding from list of floats."""
        return cls(vector=np.array(values, dtype=np.float64))

    def to_list(self) -> list[float]:
        """Convert encoding to list of floats."""
        return self.vector.tolist()

    def distance(self, other: "FaceEncoding") -> float:
        """Calculate Euclidean distance to another encoding."""
        return float(np.linalg.norm(self.vector - other.vector))

    def is_match(self, other: "FaceEncoding", tolerance: float = 0.6) -> bool:
        """Check if two encodings match within tolerance."""
        return self.distance(other) <= tolerance


class DetectedFace(BaseModel):
    """A detected face with bounding box and optional encoding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bounding_box: BoundingBox
    encoding: Optional[FaceEncoding] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    landmarks: Optional[dict[str, tuple[int, int]]] = None


class Person(BaseModel):
    """A registered person in the recognition system."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    encoding: FaceEncoding
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MatchResult(BaseModel):
    """Result of matching a face against a registered person."""

    person: Person
    distance: float = Field(..., ge=0.0)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @property
    def is_confident_match(self) -> bool:
        """Check if this is a high-confidence match (>80%)."""
        return self.confidence >= 0.8


class RecognitionResult(BaseModel):
    """Complete recognition result for a detected face."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    face: DetectedFace
    match: Optional[MatchResult] = None
    recognized: bool = False

    @classmethod
    def unknown(cls, face: DetectedFace) -> "RecognitionResult":
        """Create result for an unknown/unmatched face."""
        return cls(face=face, match=None, recognized=False)

    @classmethod
    def matched(cls, face: DetectedFace, match: MatchResult) -> "RecognitionResult":
        """Create result for a matched face."""
        return cls(face=face, match=match, recognized=True)
