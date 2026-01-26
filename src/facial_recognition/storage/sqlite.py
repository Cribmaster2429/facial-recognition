"""SQLite storage backend with async support."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

import aiosqlite

from facial_recognition.config import get_settings
from facial_recognition.core.models import FaceEncoding, Person
from facial_recognition.storage.base import BaseStorage


class SQLiteStorage(BaseStorage):
    """
    SQLite storage backend with async support.

    Persists face encodings and person data to a SQLite database file.
    Supports full CRUD operations with efficient queries.
    """

    def __init__(self, database_path: str | Path | None = None) -> None:
        """
        Initialize SQLite storage.

        Args:
            database_path: Path to SQLite database file.
                          Defaults to config setting.
        """
        if database_path is None:
            db_url = get_settings().database_url
            database_path = db_url.replace("sqlite+aiosqlite:///", "")

        self.database_path = Path(database_path)
        self._connection: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(str(self.database_path))
        self._connection.row_factory = aiosqlite.Row

        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                encoding TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name)
        """)

        await self._connection.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def save_person(self, person: Person) -> Person:
        """Save a person to the database."""
        if not self._connection:
            raise RuntimeError("Storage not initialized")

        await self._connection.execute(
            """
            INSERT OR REPLACE INTO persons
            (id, name, encoding, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(person.id),
                person.name,
                json.dumps(person.encoding.to_list()),
                json.dumps(person.metadata),
                person.created_at.isoformat(),
                person.updated_at.isoformat(),
            ),
        )
        await self._connection.commit()
        return person

    async def get_person(self, person_id: str) -> Optional[Person]:
        """Retrieve a person by ID."""
        if not self._connection:
            raise RuntimeError("Storage not initialized")

        async with self._connection.execute(
            "SELECT * FROM persons WHERE id = ?",
            (person_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return self._row_to_person(row)

    async def get_all_persons(self) -> list[Person]:
        """Retrieve all registered persons."""
        if not self._connection:
            raise RuntimeError("Storage not initialized")

        async with self._connection.execute(
            "SELECT * FROM persons ORDER BY name"
        ) as cursor:
            rows = await cursor.fetchall()

        return [self._row_to_person(row) for row in rows]

    async def delete_person(self, person_id: str) -> bool:
        """Delete a person from the database."""
        if not self._connection:
            raise RuntimeError("Storage not initialized")

        cursor = await self._connection.execute(
            "DELETE FROM persons WHERE id = ?",
            (person_id,),
        )
        await self._connection.commit()

        return cursor.rowcount > 0

    async def update_person(self, person: Person) -> Optional[Person]:
        """Update an existing person."""
        if not self._connection:
            raise RuntimeError("Storage not initialized")

        existing = await self.get_person(str(person.id))
        if existing is None:
            return None

        person.updated_at = datetime.utcnow()

        await self._connection.execute(
            """
            UPDATE persons
            SET name = ?, encoding = ?, metadata = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                person.name,
                json.dumps(person.encoding.to_list()),
                json.dumps(person.metadata),
                person.updated_at.isoformat(),
                str(person.id),
            ),
        )
        await self._connection.commit()

        return person

    async def find_by_name(self, name: str) -> list[Person]:
        """Find persons by name (case-insensitive partial match)."""
        if not self._connection:
            raise RuntimeError("Storage not initialized")

        async with self._connection.execute(
            "SELECT * FROM persons WHERE LOWER(name) LIKE ? ORDER BY name",
            (f"%{name.lower()}%",),
        ) as cursor:
            rows = await cursor.fetchall()

        return [self._row_to_person(row) for row in rows]

    async def count(self) -> int:
        """Get total count of registered persons."""
        if not self._connection:
            raise RuntimeError("Storage not initialized")

        async with self._connection.execute(
            "SELECT COUNT(*) FROM persons"
        ) as cursor:
            row = await cursor.fetchone()

        return row[0] if row else 0

    async def clear(self) -> int:
        """Remove all persons from storage."""
        if not self._connection:
            raise RuntimeError("Storage not initialized")

        count = await self.count()

        await self._connection.execute("DELETE FROM persons")
        await self._connection.commit()

        return count

    def _row_to_person(self, row: aiosqlite.Row) -> Person:
        """Convert a database row to a Person object."""
        return Person(
            id=UUID(row["id"]),
            name=row["name"],
            encoding=FaceEncoding.from_list(json.loads(row["encoding"])),
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
