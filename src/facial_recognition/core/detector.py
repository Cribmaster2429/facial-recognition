"""Face detection module using face_recognition library."""

from pathlib import Path
from typing import Literal, Union

import cv2
import face_recognition
import numpy as np
from PIL import Image

from facial_recognition.config import get_settings
from facial_recognition.core.models import BoundingBox, DetectedFace


class FaceDetector:
    """
    Face detector using HOG or CNN-based detection.

    The detector identifies faces in images and returns their bounding boxes.
    Supports both fast HOG-based detection (CPU) and accurate CNN-based
    detection (GPU recommended).
    """

    def __init__(
        self,
        model: Literal["hog", "cnn"] | None = None,
        min_face_size: int | None = None,
        number_of_times_to_upsample: int = 1,
    ) -> None:
        """
        Initialize the face detector.

        Args:
            model: Detection model - 'hog' (faster, CPU) or 'cnn' (accurate, GPU).
                   Defaults to config setting.
            min_face_size: Minimum face size in pixels. Defaults to config setting.
            number_of_times_to_upsample: How many times to upsample image for
                                         detecting smaller faces. Higher = slower.
        """
        settings = get_settings()
        self.model = model or settings.detection_model
        self.min_face_size = min_face_size or settings.min_face_size
        self.number_of_times_to_upsample = number_of_times_to_upsample

    def detect(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
    ) -> list[DetectedFace]:
        """
        Detect faces in an image.

        Args:
            image: Input image as numpy array (BGR/RGB), PIL Image, or file path.

        Returns:
            List of DetectedFace objects with bounding boxes.

        Raises:
            FileNotFoundError: If image path doesn't exist.
            ValueError: If image format is invalid.
        """
        rgb_image = self._load_image(image)

        face_locations = face_recognition.face_locations(
            rgb_image,
            model=self.model,
            number_of_times_to_upsample=self.number_of_times_to_upsample,
        )

        detected_faces = []
        for top, right, bottom, left in face_locations:
            bbox = BoundingBox(top=top, right=right, bottom=bottom, left=left)

            if bbox.width >= self.min_face_size and bbox.height >= self.min_face_size:
                detected_faces.append(DetectedFace(bounding_box=bbox))

        return detected_faces

    def detect_with_landmarks(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
    ) -> list[DetectedFace]:
        """
        Detect faces with facial landmarks.

        Args:
            image: Input image as numpy array, PIL Image, or file path.

        Returns:
            List of DetectedFace objects with bounding boxes and landmarks.
        """
        rgb_image = self._load_image(image)

        face_locations = face_recognition.face_locations(
            rgb_image,
            model=self.model,
            number_of_times_to_upsample=self.number_of_times_to_upsample,
        )

        face_landmarks_list = face_recognition.face_landmarks(
            rgb_image,
            face_locations=face_locations,
        )

        detected_faces = []
        for (top, right, bottom, left), landmarks in zip(
            face_locations, face_landmarks_list
        ):
            bbox = BoundingBox(top=top, right=right, bottom=bottom, left=left)

            if bbox.width >= self.min_face_size and bbox.height >= self.min_face_size:
                landmark_points = {}
                for feature, points in landmarks.items():
                    if points:
                        landmark_points[feature] = points[0]

                detected_faces.append(
                    DetectedFace(bounding_box=bbox, landmarks=landmark_points)
                )

        return detected_faces

    def detect_from_video_frame(
        self,
        frame: np.ndarray,
        scale: float = 0.25,
    ) -> list[DetectedFace]:
        """
        Detect faces in a video frame with optional downscaling for performance.

        Args:
            frame: Video frame as BGR numpy array (from OpenCV).
            scale: Scale factor for processing (0.25 = 4x smaller, faster).

        Returns:
            List of DetectedFace objects with bounding boxes in original coordinates.
        """
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(
            rgb_small_frame,
            model=self.model,
        )

        detected_faces = []
        inv_scale = 1.0 / scale

        for top, right, bottom, left in face_locations:
            scaled_bbox = BoundingBox(
                top=int(top * inv_scale),
                right=int(right * inv_scale),
                bottom=int(bottom * inv_scale),
                left=int(left * inv_scale),
            )

            if (
                scaled_bbox.width >= self.min_face_size
                and scaled_bbox.height >= self.min_face_size
            ):
                detected_faces.append(DetectedFace(bounding_box=scaled_bbox))

        return detected_faces

    def _load_image(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
    ) -> np.ndarray:
        """
        Load and convert image to RGB numpy array.

        Args:
            image: Input image in various formats.

        Returns:
            RGB numpy array.

        Raises:
            FileNotFoundError: If image path doesn't exist.
            ValueError: If image format is invalid.
        """
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
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        raise ValueError(f"Unsupported image type: {type(image)}")
