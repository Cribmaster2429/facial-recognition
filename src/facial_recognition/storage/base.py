"""Abstract base class for storage backends."""

from abc import ABC, abstractmethod
from typing import Optional

from facial_recognition.core.models import Person


class BaseStorage(ABC):
    """
    Abstract base class for face data storage backends.

    Provides a consistent interface for persisting and retrieving
    registered persons and their face encodings.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend (create tables, connections, etc.)."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend and release resources."""
        pass

    @abstractmethod
    async def save_person(self, person: Person) -> Person:
        """
        Save a person to storage.

        Args:
            person: Person object to save.

        Returns:
            The saved Person object.
        """
        pass

    @abstractmethod
    async def get_person(self, person_id: str) -> Optional[Person]:
        """
        Retrieve a person by ID.

        Args:
            person_id: UUID of the person.

        Returns:
            Person if found, None otherwise.
        """
        pass

    @abstractmethod
    async def get_all_persons(self) -> list[Person]:
        """
        Retrieve all registered persons.

        Returns:
            List of all Person objects.
        """
        pass

    @abstractmethod
    async def delete_person(self, person_id: str) -> bool:
        """
        Delete a person from storage.

        Args:
            person_id: UUID of the person to delete.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    async def update_person(self, person: Person) -> Optional[Person]:
        """
        Update an existing person.

        Args:
            person: Person object with updated data.

        Returns:
            Updated Person if found, None otherwise.
        """
        pass

    @abstractmethod
    async def find_by_name(self, name: str) -> list[Person]:
        """
        Find persons by name (case-insensitive partial match).

        Args:
            name: Name or partial name to search.

        Returns:
            List of matching Person objects.
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """
        Get total count of registered persons.

        Returns:
            Number of persons in storage.
        """
        pass

    @abstractmethod
    async def clear(self) -> int:
        """
        Remove all persons from storage.

        Returns:
            Number of persons removed.
        """
        pass

    async def __aenter__(self) -> "BaseStorage":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
