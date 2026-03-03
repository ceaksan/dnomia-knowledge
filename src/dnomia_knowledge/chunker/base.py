"""Chunker protocol — interface for all chunker implementations."""

from __future__ import annotations

from typing import Protocol

from dnomia_knowledge.models import Chunk


class Chunker(Protocol):
    """Protocol for chunking file content into searchable pieces."""

    def chunk(self, file_path: str, content: str) -> list[Chunk]: ...
