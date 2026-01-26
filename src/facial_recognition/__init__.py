"""
Facial Recognition System

A real-time facial recognition system using deep learning for accurate
face detection, encoding, and identification.
"""

__version__ = "1.0.0"
__author__ = "Danson Wachira"

from facial_recognition.core.detector import FaceDetector
from facial_recognition.core.encoder import FaceEncoder
from facial_recognition.core.recognizer import FaceRecognizer

__all__ = ["FaceDetector", "FaceEncoder", "FaceRecognizer", "__version__"]
