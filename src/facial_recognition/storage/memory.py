"""In-memory storage backend for testing and development."""

from datetime import datetime
from typing import Optional

from facial_recognition.core.models import Person
from facial_recognition.storage.base import BaseStorage


class MemoryStorage(BaseStorage):
    """
    In-memory storage backend.

    Stores persons in a dictionary. Data is lost when the application stops.
    Useful for testing and development.
    """

    def __init__(self) -> None:
        """Initialize the in-memory storage."""
        self._persons: dict[str, Person] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the storage."""
        self._initialized = True

    async def close(self) -> None:
        """Close the storage."""
        self._initialized = False

    async def save_person(self, person: Person) -> Person:
        """Save a person to memory."""
        self._persons[str(person.id)] = person
        return person

    async def get_person(self, person_id: str) -> Optional[Person]:
        """Retrieve a person by ID."""
        return self._persons.get(person_id)

    async def get_all_persons(self) -> list[Person]:
        """Retrieve all registered persons."""
        return list(self._persons.values())

    async def delete_person(self, person_id: str) -> bool:
        """Delete a person from storage."""
        return self._persons.pop(person_id, None) is not None

    async def update_person(self, person: Person) -> Optional[Person]:
        """Update an existing person."""
        person_id = str(person.id)
        if person_id not in self._persons:
            return None

        person.updated_at = datetime.utcnow()
        self._persons[person_id] = person
        return person

    async def find_by_name(self, name: str) -> list[Person]:
        """Find persons by name (case-insensitive partial match)."""
        name_lower = name.lower()
        return [
            p for p in self._persons.values()
            if name_lower in p.name.lower()
        ]

    async def count(self) -> int:
        """Get total count of registered persons."""
        return len(self._persons)

    async def clear(self) -> int:
        """Remove all persons from storage."""
        count = len(self._persons)
        self._persons.clear()
        return count
