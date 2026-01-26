"""Face recognition module for identifying registered persons."""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from facial_recognition.config import get_settings
from facial_recognition.core.detector import FaceDetector
from facial_recognition.core.encoder import FaceEncoder
from facial_recognition.core.models import (
    DetectedFace,
    FaceEncoding,
    MatchResult,
    Person,
    RecognitionResult,
)


class FaceRecognizer:
    """
    Face recognizer for identifying persons from a database of known faces.

    Combines face detection, encoding, and matching against registered persons
    to perform complete facial recognition pipeline.
    """

    def __init__(
        self,
        detector: FaceDetector | None = None,
        encoder: FaceEncoder | None = None,
        tolerance: float | None = None,
    ) -> None:
        """
        Initialize the face recognizer.

        Args:
            detector: FaceDetector instance. Creates default if None.
            encoder: FaceEncoder instance. Creates default if None.
            tolerance: Matching tolerance. Defaults to config setting.
        """
        self.detector = detector or FaceDetector()
        self.encoder = encoder or FaceEncoder()
        self.tolerance = tolerance or get_settings().tolerance
        self._known_persons: dict[str, Person] = {}

    @property
    def known_persons(self) -> list[Person]:
        """Get list of all registered persons."""
        return list(self._known_persons.values())

    @property
    def person_count(self) -> int:
        """Get count of registered persons."""
        return len(self._known_persons)

    def register_person(
        self,
        name: str,
        image: Union[np.ndarray, Image.Image, str, Path],
        metadata: dict[str, str] | None = None,
    ) -> Person:
        """
        Register a new person from an image.

        Args:
            name: Person's name/identifier.
            image: Image containing exactly one face.
            metadata: Optional metadata dictionary.

        Returns:
            The registered Person object.

        Raises:
            ValueError: If image contains zero or multiple faces.
        """
        faces = self.detector.detect(image)

        if len(faces) == 0:
            raise ValueError("No face detected in the image")
        if len(faces) > 1:
            raise ValueError(
                f"Multiple faces ({len(faces)}) detected. "
                "Please provide an image with exactly one face."
            )

        encoding = self.encoder.encode_face(image, faces[0])
        if encoding is None:
            raise ValueError("Failed to generate face encoding")

        person = Person(
            name=name,
            encoding=encoding,
            metadata=metadata or {},
        )

        self._known_persons[str(person.id)] = person
        return person

    def register_person_with_encoding(
        self,
        name: str,
        encoding: FaceEncoding,
        metadata: dict[str, str] | None = None,
    ) -> Person:
        """
        Register a person with a pre-computed encoding.

        Args:
            name: Person's name/identifier.
            encoding: Pre-computed face encoding.
            metadata: Optional metadata dictionary.

        Returns:
            The registered Person object.
        """
        person = Person(
            name=name,
            encoding=encoding,
            metadata=metadata or {},
        )
        self._known_persons[str(person.id)] = person
        return person

    def unregister_person(self, person_id: str) -> bool:
        """
        Remove a person from the registry.

        Args:
            person_id: UUID of the person to remove.

        Returns:
            True if person was removed, False if not found.
        """
        return self._known_persons.pop(person_id, None) is not None

    def get_person(self, person_id: str) -> Person | None:
        """Get a person by their ID."""
        return self._known_persons.get(person_id)

    def find_person_by_name(self, name: str) -> list[Person]:
        """Find persons by name (case-insensitive partial match)."""
        name_lower = name.lower()
        return [
            p for p in self._known_persons.values()
            if name_lower in p.name.lower()
        ]

    def recognize(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
    ) -> list[RecognitionResult]:
        """
        Recognize all faces in an image.

        Args:
            image: Input image containing faces to recognize.

        Returns:
            List of RecognitionResult objects, one per detected face.
        """
        faces = self.detector.detect(image)
        if not faces:
            return []

        self.encoder.encode_detected_faces(image, faces)

        results = []
        for face in faces:
            if face.encoding is None:
                results.append(RecognitionResult.unknown(face))
                continue

            match = self._find_best_match(face.encoding)
            if match:
                results.append(RecognitionResult.matched(face, match))
            else:
                results.append(RecognitionResult.unknown(face))

        return results

    def recognize_face(
        self,
        encoding: FaceEncoding,
    ) -> MatchResult | None:
        """
        Recognize a face from its encoding.

        Args:
            encoding: Face encoding to match.

        Returns:
            MatchResult if found, None otherwise.
        """
        return self._find_best_match(encoding)

    def verify(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        person_id: str,
    ) -> tuple[bool, float]:
        """
        Verify if an image contains a specific person.

        Args:
            image: Input image containing a face.
            person_id: ID of the person to verify against.

        Returns:
            Tuple of (is_verified, confidence).

        Raises:
            ValueError: If person not found or no face in image.
        """
        person = self.get_person(person_id)
        if person is None:
            raise ValueError(f"Person not found: {person_id}")

        faces = self.detector.detect(image)
        if not faces:
            raise ValueError("No face detected in the image")

        self.encoder.encode_detected_faces(image, faces)

        best_distance = float("inf")
        for face in faces:
            if face.encoding is not None:
                distance = person.encoding.distance(face.encoding)
                best_distance = min(best_distance, distance)

        is_verified = best_distance <= self.tolerance
        confidence = max(0.0, 1.0 - (best_distance / self.tolerance)) if is_verified else 0.0

        return is_verified, confidence

    def _find_best_match(self, encoding: FaceEncoding) -> MatchResult | None:
        """Find the best matching person for an encoding."""
        if not self._known_persons:
            return None

        best_match: MatchResult | None = None
        best_distance = float("inf")

        for person in self._known_persons.values():
            distance = person.encoding.distance(encoding)

            if distance <= self.tolerance and distance < best_distance:
                best_distance = distance
                confidence = 1.0 - (distance / self.tolerance)
                best_match = MatchResult(
                    person=person,
                    distance=distance,
                    confidence=confidence,
                )

        return best_match

    def clear_registry(self) -> int:
        """
        Clear all registered persons.

        Returns:
            Number of persons removed.
        """
        count = len(self._known_persons)
        self._known_persons.clear()
        return count

    def load_persons(self, persons: list[Person]) -> int:
        """
        Load multiple persons into the registry.

        Args:
            persons: List of Person objects to register.

        Returns:
            Number of persons loaded.
        """
        for person in persons:
            self._known_persons[str(person.id)] = person
        return len(persons)
