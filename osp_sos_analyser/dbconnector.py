from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import duckdb

from .db import ensure_schema, insert_commands, insert_logs
from .models import CommandArtifact, LogEntry


def _resolve_db_path(db_path: str | os.PathLike[str]) -> Path:
    raw_path = Path(db_path)
    if raw_path.is_absolute():
        return raw_path.resolve()

    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    candidates.append(cwd / raw_path)
    candidates.append((Path(__file__).resolve().parent.parent / raw_path).resolve())

    for parent in [cwd, *cwd.parents]:
        candidates.append((parent / raw_path).resolve())
        if (parent / "pyproject.toml").exists():
            candidates.append((parent / raw_path).resolve())
            break

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


class DatabaseConnector:
    """Small convenience wrapper around a DuckDB connection.

    The class exposes the most common database operations used by the project:
    opening and closing a connection, executing SQL, fetching rows, and using
    the existing ingestion helpers for logs and commands.
    """

    def __init__(
        self,
        db_path: str | os.PathLike[str],
        *,
        read_only: bool = False,
    ) -> None:
        self.db_path = _resolve_db_path(db_path)
        self.read_only = read_only
        self._connection: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            self._connection = duckdb.connect(str(self.db_path), read_only=self.read_only)
        return self._connection

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self.connect()

    def __enter__(self) -> "DatabaseConnector":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def ensure_schema(self) -> None:
        ensure_schema(self.connect())

    def insert_logs(self, entries: list[LogEntry]) -> int:
        return insert_logs(self.connect(), entries)

    def insert_commands(self, artifacts: list[CommandArtifact]) -> int:
        return insert_commands(self.connect(), artifacts)

    def execute(self, query: str, params: tuple[Any, ...] | None = None) -> duckdb.DuckDBPyConnection:
        return self.connect().execute(query, params or ())

    def fetch_all(self, query: str, params: tuple[Any, ...] | None = None) -> list[tuple[Any, ...]]:
        return self.execute(query, params).fetchall()

    def fetch_one(self, query: str, params: tuple[Any, ...] | None = None) -> tuple[Any, ...] | None:
        return self.execute(query, params).fetchone()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
