"""Tests for data models."""

import numpy as np
import pytest

from facial_recognition.core.models import (
    BoundingBox,
    DetectedFace,
    FaceEncoding,
    MatchResult,
    Person,
    RecognitionResult,
)


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_create_bounding_box(self) -> None:
        """Test creating a bounding box."""
        bbox = BoundingBox(top=10, right=110, bottom=110, left=10)

        assert bbox.top == 10
        assert bbox.right == 110
        assert bbox.bottom == 110
        assert bbox.left == 10

    def test_width_height(self) -> None:
        """Test width and height calculations."""
        bbox = BoundingBox(top=0, right=100, bottom=50, left=0)

        assert bbox.width == 100
        assert bbox.height == 50

    def test_area(self) -> None:
        """Test area calculation."""
        bbox = BoundingBox(top=0, right=100, bottom=50, left=0)

        assert bbox.area == 5000

    def test_center(self) -> None:
        """Test center point calculation."""
        bbox = BoundingBox(top=0, right=100, bottom=100, left=0)

        assert bbox.center == (50, 50)

    def test_to_tuple(self) -> None:
        """Test conversion to tuple."""
        bbox = BoundingBox(top=10, right=100, bottom=80, left=20)

        assert bbox.to_tuple() == (10, 100, 80, 20)

    def test_to_cv2_rect(self) -> None:
        """Test conversion to OpenCV rectangle format."""
        bbox = BoundingBox(top=10, right=100, bottom=80, left=20)

        assert bbox.to_cv2_rect() == (20, 10, 80, 70)


class TestFaceEncoding:
    """Tests for FaceEncoding model."""

    def test_create_encoding(self) -> None:
        """Test creating a face encoding."""
        vector = np.random.randn(128)
        encoding = FaceEncoding(vector=vector)

        assert encoding.vector.shape == (128,)

    def test_from_list(self) -> None:
        """Test creating encoding from list."""
        values = [0.1] * 128
        encoding = FaceEncoding.from_list(values)

        assert encoding.vector.shape == (128,)
        assert encoding.vector[0] == pytest.approx(0.1)

    def test_to_list(self) -> None:
        """Test converting encoding to list."""
        values = [0.1] * 128
        encoding = FaceEncoding.from_list(values)

        result = encoding.to_list()

        assert isinstance(result, list)
        assert len(result) == 128

    def test_distance_same(self) -> None:
        """Test distance between identical encodings."""
        vector = np.random.randn(128)
        enc1 = FaceEncoding(vector=vector)
        enc2 = FaceEncoding(vector=vector.copy())

        assert enc1.distance(enc2) == pytest.approx(0.0)

    def test_distance_different(self) -> None:
        """Test distance between different encodings."""
        enc1 = FaceEncoding(vector=np.ones(128))
        enc2 = FaceEncoding(vector=-np.ones(128))

        distance = enc1.distance(enc2)

        assert distance > 0

    def test_is_match_true(self) -> None:
        """Test is_match returns True for similar encodings."""
        vector = np.ones(128) * 0.5
        enc1 = FaceEncoding(vector=vector)
        enc2 = FaceEncoding(vector=vector + np.ones(128) * 0.01)

        assert enc1.is_match(enc2, tolerance=0.6)

    def test_is_match_false(self) -> None:
        """Test is_match returns False for different encodings."""
        enc1 = FaceEncoding(vector=np.ones(128))
        enc2 = FaceEncoding(vector=-np.ones(128))

        assert not enc1.is_match(enc2, tolerance=0.6)


class TestPerson:
    """Tests for Person model."""

    def test_create_person(self, sample_encoding: FaceEncoding) -> None:
        """Test creating a person."""
        person = Person(
            name="John Doe",
            encoding=sample_encoding,
        )

        assert person.name == "John Doe"
        assert person.id is not None
        assert person.created_at is not None

    def test_person_with_metadata(self, sample_encoding: FaceEncoding) -> None:
        """Test creating a person with metadata."""
        person = Person(
            name="Jane Doe",
            encoding=sample_encoding,
            metadata={"role": "admin"},
        )

        assert person.metadata["role"] == "admin"


class TestDetectedFace:
    """Tests for DetectedFace model."""

    def test_create_detected_face(self) -> None:
        """Test creating a detected face."""
        bbox = BoundingBox(top=10, right=100, bottom=100, left=10)
        face = DetectedFace(bounding_box=bbox)

        assert face.bounding_box == bbox
        assert face.encoding is None
        assert face.confidence == 1.0

    def test_detected_face_with_encoding(
        self, sample_encoding: FaceEncoding
    ) -> None:
        """Test detected face with encoding."""
        bbox = BoundingBox(top=10, right=100, bottom=100, left=10)
        face = DetectedFace(bounding_box=bbox, encoding=sample_encoding)

        assert face.encoding is not None


class TestMatchResult:
    """Tests for MatchResult model."""

    def test_create_match_result(self, sample_person: Person) -> None:
        """Test creating a match result."""
        match = MatchResult(
            person=sample_person,
            distance=0.3,
            confidence=0.85,
        )

        assert match.person == sample_person
        assert match.distance == 0.3
        assert match.confidence == 0.85

    def test_is_confident_match_true(self, sample_person: Person) -> None:
        """Test confident match detection."""
        match = MatchResult(person=sample_person, distance=0.2, confidence=0.9)

        assert match.is_confident_match

    def test_is_confident_match_false(self, sample_person: Person) -> None:
        """Test non-confident match detection."""
        match = MatchResult(person=sample_person, distance=0.5, confidence=0.7)

        assert not match.is_confident_match


class TestRecognitionResult:
    """Tests for RecognitionResult model."""

    def test_unknown_result(self) -> None:
        """Test creating unknown recognition result."""
        bbox = BoundingBox(top=10, right=100, bottom=100, left=10)
        face = DetectedFace(bounding_box=bbox)

        result = RecognitionResult.unknown(face)

        assert result.face == face
        assert result.match is None
        assert not result.recognized

    def test_matched_result(self, sample_person: Person) -> None:
        """Test creating matched recognition result."""
        bbox = BoundingBox(top=10, right=100, bottom=100, left=10)
        face = DetectedFace(bounding_box=bbox)
        match = MatchResult(person=sample_person, distance=0.3, confidence=0.85)

        result = RecognitionResult.matched(face, match)

        assert result.face == face
        assert result.match == match
        assert result.recognized
