"""Shared data models for dnomia-knowledge."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ChunkType(StrEnum):
    BLOCK = "block"
    MODULE = "module"
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    STRUCT = "struct"
    INTERFACE = "interface"
    ENUM = "enum"
    IMPL = "impl"
    TYPE = "type"
    HEADING = "heading"
    SCRIPT = "script"
    STYLE = "style"


class InteractionType(StrEnum):
    READ = "read"
    EDIT = "edit"
    SEARCH_HIT = "search_hit"


class Chunk(BaseModel):
    """A chunk of content or code extracted from a file."""

    content: str
    chunk_type: str
    name: str | None = None
    language: str | None = None
    start_line: int = 0
    end_line: int = 0
    metadata: dict | None = None  # frontmatter, tags, categories, imports


class SearchResult(BaseModel):
    """A single search result."""

    chunk_id: int
    project_id: str
    file_path: str
    chunk_domain: str  # "content" | "code"
    chunk_type: str
    name: str | None = None
    language: str | None = None
    start_line: int = 0
    end_line: int = 0
    score: float = 0.0
    snippet: str = ""


class IndexResult(BaseModel):
    """Result of an indexing operation."""

    project_id: str
    total_files: int
    indexed_files: int
    total_chunks: int
    content_chunks: int = 0
    code_chunks: int = 0
    duration_seconds: float = 0.0
    status: str = "completed"
