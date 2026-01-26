"""Tests for storage backends."""

from pathlib import Path

import pytest

from facial_recognition.core.models import FaceEncoding, Person
from facial_recognition.storage.memory import MemoryStorage
from facial_recognition.storage.sqlite import SQLiteStorage


class TestMemoryStorage:
    """Tests for MemoryStorage backend."""

    @pytest.mark.asyncio
    async def test_save_and_get_person(
        self, memory_storage: MemoryStorage, sample_person: Person
    ) -> None:
        """Test saving and retrieving a person."""
        await memory_storage.save_person(sample_person)
        retrieved = await memory_storage.get_person(str(sample_person.id))

        assert retrieved is not None
        assert retrieved.id == sample_person.id
        assert retrieved.name == sample_person.name

    @pytest.mark.asyncio
    async def test_get_nonexistent_person(
        self, memory_storage: MemoryStorage
    ) -> None:
        """Test retrieving a non-existent person."""
        result = await memory_storage.get_person("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_persons(
        self, memory_storage: MemoryStorage, sample_encoding: FaceEncoding
    ) -> None:
        """Test retrieving all persons."""
        person1 = Person(name="Alice", encoding=sample_encoding)
        person2 = Person(name="Bob", encoding=sample_encoding)

        await memory_storage.save_person(person1)
        await memory_storage.save_person(person2)

        persons = await memory_storage.get_all_persons()

        assert len(persons) == 2

    @pytest.mark.asyncio
    async def test_delete_person(
        self, memory_storage: MemoryStorage, sample_person: Person
    ) -> None:
        """Test deleting a person."""
        await memory_storage.save_person(sample_person)
        deleted = await memory_storage.delete_person(str(sample_person.id))

        assert deleted is True
        assert await memory_storage.get_person(str(sample_person.id)) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_person(
        self, memory_storage: MemoryStorage
    ) -> None:
        """Test deleting a non-existent person."""
        deleted = await memory_storage.delete_person("nonexistent-id")

        assert deleted is False

    @pytest.mark.asyncio
    async def test_update_person(
        self, memory_storage: MemoryStorage, sample_person: Person
    ) -> None:
        """Test updating a person."""
        await memory_storage.save_person(sample_person)

        sample_person.name = "Updated Name"
        updated = await memory_storage.update_person(sample_person)

        assert updated is not None
        assert updated.name == "Updated Name"

    @pytest.mark.asyncio
    async def test_find_by_name(
        self, memory_storage: MemoryStorage, sample_encoding: FaceEncoding
    ) -> None:
        """Test finding persons by name."""
        person1 = Person(name="Alice Smith", encoding=sample_encoding)
        person2 = Person(name="Bob Jones", encoding=sample_encoding)

        await memory_storage.save_person(person1)
        await memory_storage.save_person(person2)

        results = await memory_storage.find_by_name("alice")

        assert len(results) == 1
        assert results[0].name == "Alice Smith"

    @pytest.mark.asyncio
    async def test_count(
        self, memory_storage: MemoryStorage, sample_encoding: FaceEncoding
    ) -> None:
        """Test counting persons."""
        assert await memory_storage.count() == 0

        await memory_storage.save_person(Person(name="Test", encoding=sample_encoding))

        assert await memory_storage.count() == 1

    @pytest.mark.asyncio
    async def test_clear(
        self, memory_storage: MemoryStorage, sample_encoding: FaceEncoding
    ) -> None:
        """Test clearing all persons."""
        await memory_storage.save_person(Person(name="Test", encoding=sample_encoding))

        count = await memory_storage.clear()

        assert count == 1
        assert await memory_storage.count() == 0


class TestSQLiteStorage:
    """Tests for SQLiteStorage backend."""

    @pytest.mark.asyncio
    async def test_initialize(self, temp_db_path: Path) -> None:
        """Test storage initialization."""
        storage = SQLiteStorage(temp_db_path)
        await storage.initialize()

        assert temp_db_path.exists()

        await storage.close()

    @pytest.mark.asyncio
    async def test_save_and_get_person(
        self, temp_db_path: Path, sample_person: Person
    ) -> None:
        """Test saving and retrieving a person."""
        async with SQLiteStorage(temp_db_path) as storage:
            await storage.save_person(sample_person)
            retrieved = await storage.get_person(str(sample_person.id))

        assert retrieved is not None
        assert retrieved.id == sample_person.id
        assert retrieved.name == sample_person.name

    @pytest.mark.asyncio
    async def test_persistence(
        self, temp_db_path: Path, sample_person: Person
    ) -> None:
        """Test that data persists across connections."""
        async with SQLiteStorage(temp_db_path) as storage:
            await storage.save_person(sample_person)

        async with SQLiteStorage(temp_db_path) as storage:
            retrieved = await storage.get_person(str(sample_person.id))

        assert retrieved is not None
        assert retrieved.name == sample_person.name

    @pytest.mark.asyncio
    async def test_encoding_persistence(
        self, temp_db_path: Path, sample_person: Person
    ) -> None:
        """Test that encodings are correctly persisted."""
        async with SQLiteStorage(temp_db_path) as storage:
            await storage.save_person(sample_person)
            retrieved = await storage.get_person(str(sample_person.id))

        assert retrieved is not None
        distance = sample_person.encoding.distance(retrieved.encoding)
        assert distance < 0.001

    @pytest.mark.asyncio
    async def test_context_manager(self, temp_db_path: Path) -> None:
        """Test async context manager."""
        async with SQLiteStorage(temp_db_path) as storage:
            count = await storage.count()
            assert count == 0
