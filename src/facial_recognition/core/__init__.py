"""Core facial recognition modules."""

from facial_recognition.core.detector import FaceDetector
from facial_recognition.core.encoder import FaceEncoder
from facial_recognition.core.models import (
    BoundingBox,
    DetectedFace,
    FaceEncoding,
    MatchResult,
    Person,
    RecognitionResult,
)
from facial_recognition.core.recognizer import FaceRecognizer

__all__ = [
    "FaceDetector",
    "FaceEncoder",
    "FaceRecognizer",
    "BoundingBox",
    "DetectedFace",
    "FaceEncoding",
    "MatchResult",
    "Person",
    "RecognitionResult",
]
