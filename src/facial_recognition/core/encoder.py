"""Face encoding module for generating face embeddings."""

from pathlib import Path
from typing import Literal, Union

import cv2
import face_recognition
import numpy as np
from PIL import Image

from facial_recognition.config import get_settings
from facial_recognition.core.models import (
    BoundingBox,
    DetectedFace,
    FaceEncoding,
)


class FaceEncoder:
    """
    Face encoder for generating 128-dimensional face embeddings.

    Uses a deep neural network to create face encodings that can be compared
    for face matching and recognition. The encodings are robust to lighting,
    pose variations, and minor occlusions.
    """

    def __init__(
        self,
        model: Literal["small", "large"] | None = None,
        num_jitters: int = 1,
    ) -> None:
        """
        Initialize the face encoder.

        Args:
            model: Encoding model - 'small' (faster, 5 landmarks) or
                   'large' (accurate, 68 landmarks). Defaults to config setting.
            num_jitters: Number of times to re-sample face for encoding.
                        Higher = more accurate but slower.
        """
        settings = get_settings()
        self.model = model or settings.encoding_model
        self.num_jitters = num_jitters

    def encode(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        face_locations: list[BoundingBox] | None = None,
    ) -> list[FaceEncoding]:
        """
        Generate face encodings for detected faces in an image.

        Args:
            image: Input image as numpy array, PIL Image, or file path.
            face_locations: Optional pre-computed face bounding boxes.
                           If None, faces will be detected automatically.

        Returns:
            List of FaceEncoding objects, one per detected face.
        """
        rgb_image = self._load_image(image)

        locations = None
        if face_locations:
            locations = [bbox.to_tuple() for bbox in face_locations]

        raw_encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=locations,
            num_jitters=self.num_jitters,
            model=self.model,
        )

        return [FaceEncoding(vector=enc) for enc in raw_encodings]

    def encode_face(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        face: DetectedFace,
    ) -> FaceEncoding | None:
        """
        Generate encoding for a single detected face.

        Args:
            image: Source image containing the face.
            face: DetectedFace object with bounding box.

        Returns:
            FaceEncoding if successful, None if encoding fails.
        """
        encodings = self.encode(image, [face.bounding_box])
        return encodings[0] if encodings else None

    def encode_detected_faces(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        faces: list[DetectedFace],
    ) -> list[DetectedFace]:
        """
        Add encodings to a list of detected faces.

        Args:
            image: Source image containing the faces.
            faces: List of DetectedFace objects to encode.

        Returns:
            Same list of DetectedFace objects with encodings populated.
        """
        if not faces:
            return faces

        rgb_image = self._load_image(image)
        locations = [face.bounding_box.to_tuple() for face in faces]

        raw_encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=locations,
            num_jitters=self.num_jitters,
            model=self.model,
        )

        for face, encoding in zip(faces, raw_encodings):
            face.encoding = FaceEncoding(vector=encoding)

        return faces

    def encode_from_video_frame(
        self,
        frame: np.ndarray,
        faces: list[DetectedFace],
    ) -> list[DetectedFace]:
        """
        Encode faces from a video frame (BGR format from OpenCV).

        Args:
            frame: Video frame as BGR numpy array.
            faces: List of DetectedFace objects to encode.

        Returns:
            DetectedFace objects with encodings populated.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.encode_detected_faces(rgb_frame, faces)

    def compare_encodings(
        self,
        known_encoding: FaceEncoding,
        unknown_encoding: FaceEncoding,
        tolerance: float | None = None,
    ) -> tuple[bool, float]:
        """
        Compare two face encodings.

        Args:
            known_encoding: The reference face encoding.
            unknown_encoding: The face encoding to compare.
            tolerance: Maximum distance for a match. Defaults to config setting.

        Returns:
            Tuple of (is_match, distance).
        """
        if tolerance is None:
            tolerance = get_settings().tolerance

        distance = known_encoding.distance(unknown_encoding)
        is_match = distance <= tolerance

        return is_match, distance

    def batch_compare(
        self,
        known_encodings: list[FaceEncoding],
        unknown_encoding: FaceEncoding,
        tolerance: float | None = None,
    ) -> list[tuple[int, float]]:
        """
        Compare an unknown encoding against multiple known encodings.

        Args:
            known_encodings: List of reference face encodings.
            unknown_encoding: The face encoding to compare.
            tolerance: Maximum distance for a match. Defaults to config setting.

        Returns:
            List of (index, distance) tuples for matches, sorted by distance.
        """
        if tolerance is None:
            tolerance = get_settings().tolerance

        matches = []
        for i, known in enumerate(known_encodings):
            distance = known.distance(unknown_encoding)
            if distance <= tolerance:
                matches.append((i, distance))

        return sorted(matches, key=lambda x: x[1])

    def _load_image(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
    ) -> np.ndarray:
        """Load and convert image to RGB numpy array."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return face_recognition.load_image_file(str(path))

        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            if image.shape[2] == 3:
                return image
            return image

        raise ValueError(f"Unsupported image type: {type(image)}")
