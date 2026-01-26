"""Storage backends for face data persistence."""

from facial_recognition.storage.base import BaseStorage
from facial_recognition.storage.memory import MemoryStorage
from facial_recognition.storage.sqlite import SQLiteStorage

__all__ = ["BaseStorage", "MemoryStorage", "SQLiteStorage"]
