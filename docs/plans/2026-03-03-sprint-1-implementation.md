# Sprint 1: Working Skeleton — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Tek bir markdown dosyasini indeksleyip hybrid search ile arayabilen calisan bir MCP server.

**Architecture:** Pure Python MCP server. SQLite + sqlite-vec + FTS5 tek DB. multilingual-e5-base embedding. FastMCP tool definitions. Heading-based markdown chunking.

**Tech Stack:** Python 3.11+, FastMCP (mcp[cli]), sentence-transformers, sqlite-vec, FTS5, Pydantic v2, pathspec, python-frontmatter

**Reference Docs:**
- Design doc: `docs/plans/2026-03-02-dnomia-knowledge-design.md`
- Reuse: `ceaksan-v4.0/.claude/knowledge/scripts/` (embedding, chunking, graph)
- Reuse: `mcp-code-search/` (AST chunking, hybrid search, RRF, file scanning)

---

### Task 0: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/dnomia_knowledge/__init__.py`
- Create: `src/dnomia_knowledge/models.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`
- Create: `.python-version`

**Step 1: Initialize git repo**

```bash
cd /Users/ceair/Desktop/dnomia-knowledge
git init
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "dnomia-knowledge"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mcp[cli]>=1.13.0",
    "sentence-transformers>=3.0.0",
    "sqlite-vec>=0.1.6",
    "pathspec>=0.12.0",
    "pydantic>=2.0.0",
    "python-frontmatter>=1.0.0",
    "pyyaml>=6.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
build/
.ruff_cache/
.pytest_cache/
*.db
*.db-wal
*.db-shm
```

**Step 4: Create .python-version**

```
3.11
```

**Step 5: Create directory structure and __init__.py files**

```bash
mkdir -p src/dnomia_knowledge/chunker tests
touch src/dnomia_knowledge/__init__.py
touch src/dnomia_knowledge/chunker/__init__.py
touch tests/__init__.py
```

**Step 6: Create models.py — shared Pydantic data models**

```python
"""Shared data models for dnomia-knowledge."""

from __future__ import annotations

from pydantic import BaseModel


class Chunk(BaseModel):
    """A chunk of content or code extracted from a file."""

    content: str
    chunk_type: str  # "heading" | "function" | "class" | "method" | "block" | "module"
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
```

**Step 7: Create tests/conftest.py**

```python
"""Shared test fixtures."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def db_path(tmp_dir):
    """Temporary database path."""
    return str(tmp_dir / "test.db")


@pytest.fixture
def sample_markdown():
    """Sample markdown content for testing."""
    return """---
title: Test Article
tags: [python, testing]
categories: [tutorial]
description: A test article about Python testing
---

## Introduction

This is a test article about Python testing. It covers the basics of pytest.

## Writing Tests

Here is how you write a test:

```python
def test_example():
    assert 1 + 1 == 2
```

Always use descriptive test names.

## Running Tests

You can run tests with:

```bash
pytest -v
```

## Conclusion

Testing is important for code quality.
"""


@pytest.fixture
def sample_markdown_file(tmp_dir, sample_markdown):
    """Write sample markdown to a file and return the path."""
    p = tmp_dir / "test-article.md"
    p.write_text(sample_markdown)
    return p
```

**Step 8: Create venv and install dependencies**

```bash
cd /Users/ceair/Desktop/dnomia-knowledge
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Step 9: Verify setup**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && python -c "import dnomia_knowledge; print('OK')"`
Expected: `OK`

**Step 10: Commit**

```bash
git add pyproject.toml .gitignore .python-version src/ tests/
git commit -m "feat: project scaffolding with pyproject.toml, models, and test fixtures"
```

---

### Task 1: store.py — SQLite Schema + CRUD

**Files:**
- Create: `src/dnomia_knowledge/store.py`
- Create: `tests/test_store.py`

**Step 1: Write the failing tests**

```python
"""Tests for SQLite store."""

from __future__ import annotations

import json

import pytest

from dnomia_knowledge.store import Store


class TestStoreInit:
    def test_creates_database(self, db_path):
        store = Store(db_path)
        assert store.db_path == db_path

    def test_creates_tables(self, db_path):
        store = Store(db_path)
        conn = store._connect()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        expected = {
            "projects",
            "file_index",
            "chunks",
            "edges",
            "search_log",
            "chunk_interactions",
            "system_metadata",
        }
        assert expected.issubset(tables)

    def test_creates_virtual_tables(self, db_path):
        store = Store(db_path)
        conn = store._connect()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'chunks_%'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "chunks_fts" in tables

    def test_sets_pragmas(self, db_path):
        store = Store(db_path)
        conn = store._connect()
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal == "wal"

    def test_sets_system_metadata(self, db_path):
        store = Store(db_path)
        conn = store._connect()
        row = conn.execute(
            "SELECT value FROM system_metadata WHERE key = 'schema_version'"
        ).fetchone()
        assert row is not None
        assert row[0] == "1"


class TestProjectCRUD:
    def test_register_project(self, db_path):
        store = Store(db_path)
        store.register_project("test-proj", "/tmp/test-proj", "content", graph_enabled=True)
        proj = store.get_project("test-proj")
        assert proj is not None
        assert proj["path"] == "/tmp/test-proj"
        assert proj["type"] == "content"
        assert proj["graph_enabled"] == 1

    def test_list_projects(self, db_path):
        store = Store(db_path)
        store.register_project("proj-a", "/tmp/a", "content")
        store.register_project("proj-b", "/tmp/b", "saas")
        projects = store.list_projects()
        assert len(projects) == 2

    def test_register_duplicate_project_updates(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        store.register_project("test", "/tmp/test", "saas")
        proj = store.get_project("test")
        assert proj["type"] == "saas"


class TestChunkCRUD:
    def test_insert_chunks(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        chunk_ids = store.insert_chunks(
            "test",
            [
                {
                    "file_path": "readme.md",
                    "chunk_domain": "content",
                    "chunk_type": "heading",
                    "name": "Introduction",
                    "language": "md",
                    "start_line": 1,
                    "end_line": 10,
                    "content": "This is the introduction.",
                    "metadata": json.dumps({"tags": ["python"]}),
                },
            ],
        )
        assert len(chunk_ids) == 1
        assert isinstance(chunk_ids[0], int)

    def test_delete_file_chunks(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        store.insert_chunks(
            "test",
            [
                {
                    "file_path": "a.md",
                    "chunk_domain": "content",
                    "chunk_type": "heading",
                    "content": "chunk a",
                },
                {
                    "file_path": "b.md",
                    "chunk_domain": "content",
                    "chunk_type": "heading",
                    "content": "chunk b",
                },
            ],
        )
        store.delete_file_chunks("test", "a.md")
        conn = store._connect()
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count == 1

    def test_get_project_stats(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        store.insert_chunks(
            "test",
            [
                {
                    "file_path": "a.md",
                    "chunk_domain": "content",
                    "chunk_type": "heading",
                    "content": "content chunk",
                },
                {
                    "file_path": "b.py",
                    "chunk_domain": "code",
                    "chunk_type": "function",
                    "content": "def foo(): pass",
                },
            ],
        )
        stats = store.get_project_stats("test")
        assert stats["total_chunks"] == 2
        assert stats["content_chunks"] == 1
        assert stats["code_chunks"] == 1


class TestFileIndex:
    def test_upsert_file_index(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        store.upsert_file_index("test", "readme.md", "abc123", 3)
        row = store.get_file_hash("test", "readme.md")
        assert row == "abc123"

    def test_get_all_file_hashes(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        store.upsert_file_index("test", "a.md", "hash_a", 2)
        store.upsert_file_index("test", "b.md", "hash_b", 1)
        hashes = store.get_all_file_hashes("test")
        assert hashes == {"a.md": "hash_a", "b.md": "hash_b"}

    def test_delete_file_index(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        store.upsert_file_index("test", "a.md", "hash_a", 2)
        store.delete_file_index("test", "a.md")
        assert store.get_file_hash("test", "a.md") is None


class TestSystemMetadata:
    def test_get_set_metadata(self, db_path):
        store = Store(db_path)
        store.set_metadata("test_key", "test_value")
        assert store.get_metadata("test_key") == "test_value"

    def test_get_missing_metadata(self, db_path):
        store = Store(db_path)
        assert store.get_metadata("nonexistent") is None

    def test_update_metadata(self, db_path):
        store = Store(db_path)
        store.set_metadata("key", "v1")
        store.set_metadata("key", "v2")
        assert store.get_metadata("key") == "v2"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && python -m pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dnomia_knowledge.store'`

**Step 3: Write store.py**

```python
"""SQLite store with FTS5 and sqlite-vec support."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA_VERSION = "1"

_PRAGMAS = [
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA temp_store = MEMORY",
    "PRAGMA mmap_size = 67108864",
    "PRAGMA cache_size = -64000",
]

_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL,
    graph_enabled INTEGER DEFAULT 0,
    last_indexed TEXT,
    config_hash TEXT
);

CREATE TABLE IF NOT EXISTS file_index (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    last_indexed TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (project_id, file_path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    chunk_domain TEXT NOT NULL DEFAULT 'content',
    chunk_type TEXT NOT NULL DEFAULT 'block',
    name TEXT,
    language TEXT,
    start_line INTEGER DEFAULT 0,
    end_line INTEGER DEFAULT 0,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS edges (
    source_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    metadata TEXT,
    PRIMARY KEY (source_id, target_id, edge_type)
);

CREATE TABLE IF NOT EXISTS search_log (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    project_id TEXT,
    domain TEXT,
    result_chunk_ids TEXT,
    result_count INTEGER DEFAULT 0,
    timestamp TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunk_interactions (
    id INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    interaction TEXT NOT NULL,
    source_tool TEXT,
    timestamp TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS system_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content=chunks,
    content_rowid=id,
    tokenize='porter unicode61'
);
"""

_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_vec_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_vec WHERE id = old.id;
END;
"""

_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project_id);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_domain ON chunks(chunk_domain);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_search_log_project ON search_log(project_id);
CREATE INDEX IF NOT EXISTS idx_search_log_ts ON search_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_interactions_chunk ON chunk_interactions(chunk_id);
CREATE INDEX IF NOT EXISTS idx_interactions_ts ON chunk_interactions(timestamp);
"""

# Embedding dimension — set by embedder on first use
_DEFAULT_EMBEDDING_DIM = 768


class Store:
    """SQLite store for chunks, embeddings, and search."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            # Load sqlite-vec extension
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            # Set performance PRAGMAs
            for pragma in _PRAGMAS:
                self._conn.execute(pragma)
        return self._conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.executescript(_TABLES_SQL)
        conn.executescript(_FTS_SQL)
        conn.executescript(_TRIGGERS_SQL)
        # vec0 table — created via sqlite_vec
        conn.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[{_DEFAULT_EMBEDDING_DIM}]
            )"""
        )
        conn.executescript(_INDEXES_SQL)
        # System metadata defaults
        conn.execute(
            "INSERT OR IGNORE INTO system_metadata (key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # -- Projects --

    def register_project(
        self,
        project_id: str,
        path: str,
        project_type: str,
        graph_enabled: bool = False,
        config_hash: str | None = None,
    ) -> None:
        conn = self._connect()
        conn.execute(
            """INSERT INTO projects (id, path, type, graph_enabled, config_hash)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   path = excluded.path,
                   type = excluded.type,
                   graph_enabled = excluded.graph_enabled,
                   config_hash = excluded.config_hash""",
            (project_id, path, project_type, int(graph_enabled), config_hash),
        )
        conn.commit()

    def get_project(self, project_id: str) -> dict | None:
        conn = self._connect()
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return dict(row) if row else None

    def list_projects(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute("SELECT * FROM projects ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    # -- Chunks --

    def insert_chunks(self, project_id: str, chunks: list[dict]) -> list[int]:
        conn = self._connect()
        ids = []
        for c in chunks:
            cursor = conn.execute(
                """INSERT INTO chunks
                   (project_id, file_path, chunk_domain, chunk_type, name,
                    language, start_line, end_line, content, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    project_id,
                    c.get("file_path", ""),
                    c.get("chunk_domain", "content"),
                    c.get("chunk_type", "block"),
                    c.get("name"),
                    c.get("language"),
                    c.get("start_line", 0),
                    c.get("end_line", 0),
                    c["content"],
                    c.get("metadata"),
                ),
            )
            ids.append(cursor.lastrowid)
        conn.commit()
        return ids

    def insert_chunk_vectors(self, chunk_ids: list[int], vectors: list[list[float]]) -> None:
        conn = self._connect()
        for chunk_id, vec in zip(chunk_ids, vectors):
            conn.execute(
                "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                (chunk_id, json.dumps(vec)),
            )
        conn.commit()

    def delete_file_chunks(self, project_id: str, file_path: str) -> int:
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM chunks WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        )
        conn.commit()
        return cursor.rowcount

    def get_project_stats(self, project_id: str) -> dict:
        conn = self._connect()
        total = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE project_id = ?", (project_id,)
        ).fetchone()[0]
        content = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE project_id = ? AND chunk_domain = 'content'",
            (project_id,),
        ).fetchone()[0]
        code = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE project_id = ? AND chunk_domain = 'code'",
            (project_id,),
        ).fetchone()[0]
        files = conn.execute(
            "SELECT COUNT(DISTINCT file_path) FROM chunks WHERE project_id = ?",
            (project_id,),
        ).fetchone()[0]
        return {
            "total_chunks": total,
            "content_chunks": content,
            "code_chunks": code,
            "total_files": files,
        }

    # -- File Index --

    def upsert_file_index(
        self, project_id: str, file_path: str, file_hash: str, chunk_count: int
    ) -> None:
        conn = self._connect()
        conn.execute(
            """INSERT INTO file_index (project_id, file_path, file_hash, chunk_count)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(project_id, file_path) DO UPDATE SET
                   file_hash = excluded.file_hash,
                   chunk_count = excluded.chunk_count,
                   last_indexed = datetime('now')""",
            (project_id, file_path, file_hash, chunk_count),
        )
        conn.commit()

    def get_file_hash(self, project_id: str, file_path: str) -> str | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT file_hash FROM file_index WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        ).fetchone()
        return row[0] if row else None

    def get_all_file_hashes(self, project_id: str) -> dict[str, str]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT file_path, file_hash FROM file_index WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def delete_file_index(self, project_id: str, file_path: str) -> None:
        conn = self._connect()
        conn.execute(
            "DELETE FROM file_index WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        )
        conn.commit()

    # -- System Metadata --

    def set_metadata(self, key: str, value: str) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT INTO system_metadata (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()

    def get_metadata(self, key: str) -> str | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT value FROM system_metadata WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None
```

**Step 4: Run tests**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && python -m pytest tests/test_store.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/dnomia_knowledge/store.py tests/test_store.py
git commit -m "feat: SQLite store with schema, CRUD, FTS5, sqlite-vec, and system metadata"
```

---

### Task 2: embedder.py — Embedding with e5 Prefix

**Files:**
- Create: `src/dnomia_knowledge/embedder.py`
- Create: `tests/test_embedder.py`

**Step 1: Write the failing tests**

```python
"""Tests for embedding module."""

from __future__ import annotations

import pytest

from dnomia_knowledge.embedder import Embedder


class TestEmbedder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.embedder = Embedder()

    def test_dimension(self):
        assert self.embedder.dimension == 768

    def test_embed_query_returns_list(self):
        result = self.embedder.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)

    def test_embed_passage_returns_list(self):
        result = self.embedder.embed_passage("test passage")
        assert isinstance(result, list)
        assert len(result) == 768

    def test_embed_passages_batch(self):
        texts = ["passage one", "passage two", "passage three"]
        results = self.embedder.embed_passages(texts)
        assert len(results) == 3
        assert all(len(v) == 768 for v in results)

    def test_query_and_passage_differ(self):
        """query: prefix and passage: prefix should produce different embeddings."""
        q = self.embedder.embed_query("python testing")
        p = self.embedder.embed_passage("python testing")
        assert q != p

    def test_query_cache(self):
        """Same query should return cached result."""
        r1 = self.embedder.embed_query("cache test")
        r2 = self.embedder.embed_query("cache test")
        assert r1 == r2

    def test_lazy_loading(self):
        """Model should not be loaded until first embed call."""
        embedder = Embedder()
        assert embedder._model is None
        embedder.embed_query("trigger load")
        assert embedder._model is not None

    def test_unload(self):
        """Model should be unloadable."""
        self.embedder.embed_query("trigger load")
        assert self.embedder._model is not None
        self.embedder.unload()
        assert self.embedder._model is None
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && python -m pytest tests/test_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write embedder.py**

```python
"""Embedding module using multilingual-e5-base with query/passage prefix."""

from __future__ import annotations

import gc
import hashlib
import time
from typing import Any

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_DIMENSION = 768


class Embedder:
    """Lazy-loading embedding with query/passage prefix and caching."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_ttl_minutes: int = 15,
    ):
        self.model_name = model_name
        self.dimension = DEFAULT_DIMENSION
        self._model: Any = None
        self._last_used: float = 0
        self._cache: dict[str, tuple[list[float], float]] = {}
        self._cache_ttl = cache_ttl_minutes * 60

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._last_used = time.time()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        self._last_used = time.time()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        cache_key = hashlib.md5(f"q:{text}".encode()).hexdigest()
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        result = self._encode([f"query: {text}"])[0]
        self._set_cached(cache_key, result)
        return result

    def embed_passage(self, text: str) -> list[float]:
        return self._encode([f"passage: {text}"])[0]

    def embed_passages(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        all_results: list[list[float]] = []
        prefixed = [f"passage: {t}" for t in texts]
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i : i + batch_size]
            all_results.extend(self._encode(batch))
        return all_results

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()

    def maybe_unload(self, idle_minutes: int = 10) -> None:
        if self._model and time.time() - self._last_used > idle_minutes * 60:
            self.unload()

    def _get_cached(self, key: str) -> list[float] | None:
        if key in self._cache:
            value, ts = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: list[float]) -> None:
        now = time.time()
        self._cache[key] = (value, now)
        # Evict expired entries
        expired = [k for k, (_, ts) in self._cache.items() if now - ts >= self._cache_ttl]
        for k in expired:
            del self._cache[k]
```

**Step 4: Run tests**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && python -m pytest tests/test_embedder.py -v`
Expected: ALL PASS (ilk calistirmada model download olacak, ~1 dakika bekle)

**Step 5: Commit**

```bash
git add src/dnomia_knowledge/embedder.py tests/test_embedder.py
git commit -m "feat: embedder with multilingual-e5-base, query/passage prefix, lazy loading, and cache"
```

---

### Task 3: md_chunker.py — Heading-Based Markdown Chunking

**Files:**
- Create: `src/dnomia_knowledge/chunker/base.py`
- Create: `src/dnomia_knowledge/chunker/md_chunker.py`
- Create: `tests/test_chunker.py`

**Step 1: Write the failing tests**

```python
"""Tests for markdown chunker."""

from __future__ import annotations

import pytest

from dnomia_knowledge.chunker.md_chunker import MdChunker
from dnomia_knowledge.models import Chunk


class TestMdChunker:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.chunker = MdChunker()

    def test_basic_heading_split(self, sample_markdown):
        chunks = self.chunker.chunk("test.md", sample_markdown)
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_types_are_heading(self, sample_markdown):
        chunks = self.chunker.chunk("test.md", sample_markdown)
        for c in chunks:
            assert c.chunk_type == "heading"

    def test_heading_names_extracted(self, sample_markdown):
        chunks = self.chunker.chunk("test.md", sample_markdown)
        names = [c.name for c in chunks]
        assert "Introduction" in names
        assert "Writing Tests" in names

    def test_frontmatter_in_metadata(self, sample_markdown):
        chunks = self.chunker.chunk("test.md", sample_markdown)
        for c in chunks:
            assert c.metadata is not None
            assert "title" in c.metadata
            assert c.metadata["title"] == "Test Article"
            assert "tags" in c.metadata

    def test_code_blocks_preserved(self, sample_markdown):
        chunks = self.chunker.chunk("test.md", sample_markdown)
        code_found = any("def test_example" in c.content for c in chunks)
        assert code_found

    def test_small_chunks_merged(self):
        content = """---
title: Test
---

## Section A

Tiny.

## Section B

Also tiny.

## Section C

This section has enough content to stand alone. It contains multiple sentences
and paragraphs of text that make it a reasonable standalone chunk for the
knowledge system to index and search through.
"""
        chunker = MdChunker(min_chunk_chars=100)
        chunks = chunker.chunk("test.md", content)
        # Tiny sections should be merged
        assert len(chunks) < 3

    def test_line_numbers(self, sample_markdown):
        chunks = self.chunker.chunk("test.md", sample_markdown)
        for c in chunks:
            assert c.start_line > 0
            assert c.end_line >= c.start_line

    def test_language_set_to_md(self, sample_markdown):
        chunks = self.chunker.chunk("test.md", sample_markdown)
        for c in chunks:
            assert c.language == "md"

    def test_mdx_language(self):
        chunks = self.chunker.chunk("test.mdx", "## Hello\n\nWorld")
        assert all(c.language == "mdx" for c in chunks)

    def test_empty_content(self):
        chunks = self.chunker.chunk("test.md", "")
        assert chunks == []

    def test_no_headings(self):
        content = "Just plain text without any headings."
        chunks = self.chunker.chunk("test.md", content)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "heading"
        assert chunks[0].name is None

    def test_overlap(self):
        content = """---
title: Test
---

## Section A

Line 1 of section A.
Line 2 of section A.
Last line of section A.

## Section B

Line 1 of section B.
"""
        chunker = MdChunker(overlap_lines=2)
        chunks = chunker.chunk("test.md", content)
        if len(chunks) >= 2:
            # Second chunk should contain overlap from first
            # (last 2 lines of section A should appear in section B's chunk)
            assert chunks[1].start_line <= chunks[0].end_line
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && python -m pytest tests/test_chunker.py -v`
Expected: FAIL

**Step 3: Write chunker/base.py**

```python
"""Chunker protocol — interface for all chunker implementations."""

from __future__ import annotations

from typing import Protocol

from dnomia_knowledge.models import Chunk


class Chunker(Protocol):
    """Protocol for chunking file content into searchable pieces."""

    def chunk(self, file_path: str, content: str) -> list[Chunk]: ...
```

**Step 4: Write chunker/md_chunker.py**

Kaynak: ceaksan-knowledge/index_content.py'den heading split, frontmatter parse, small chunk merge mantigi alinacak.

```python
"""Heading-based markdown/MDX chunker with frontmatter support."""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter

from dnomia_knowledge.models import Chunk

# Heading pattern: ## and ### only (# is title, #### too granular)
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)

# Default min chars for a chunk (below this, merge into previous)
_DEFAULT_MIN_CHUNK_CHARS = 200


class MdChunker:
    """Chunk markdown/MDX files by heading boundaries."""

    def __init__(
        self,
        overlap_lines: int = 0,
        min_chunk_chars: int = _DEFAULT_MIN_CHUNK_CHARS,
    ):
        self.overlap_lines = overlap_lines
        self.min_chunk_chars = min_chunk_chars

    def chunk(self, file_path: str, content: str) -> list[Chunk]:
        if not content or not content.strip():
            return []

        ext = Path(file_path).suffix.lstrip(".")
        language = ext if ext in ("md", "mdx") else "md"

        # Parse frontmatter
        fm_meta = self._parse_frontmatter(content)
        body = self._strip_frontmatter(content)

        if not body.strip():
            return []

        # Find heading positions
        lines = body.split("\n")
        sections = self._split_by_headings(lines)

        if not sections:
            return [
                Chunk(
                    content=body.strip(),
                    chunk_type="heading",
                    name=None,
                    language=language,
                    start_line=1,
                    end_line=len(content.split("\n")),
                    metadata=fm_meta,
                )
            ]

        # Merge small sections
        merged = self._merge_small(sections)

        # Apply overlap
        if self.overlap_lines > 0:
            merged = self._apply_overlap(merged, lines)

        # Build Chunk objects
        # Calculate frontmatter offset (lines before body starts)
        fm_lines = len(content.split("\n")) - len(lines)

        chunks = []
        for section in merged:
            chunks.append(
                Chunk(
                    content=section["text"].strip(),
                    chunk_type="heading",
                    name=section.get("heading"),
                    language=language,
                    start_line=section["start_line"] + fm_lines,
                    end_line=section["end_line"] + fm_lines,
                    metadata=fm_meta,
                )
            )
        return chunks

    def _parse_frontmatter(self, content: str) -> dict | None:
        try:
            post = frontmatter.loads(content)
            meta = dict(post.metadata)
            if not meta:
                return None
            # Keep only useful fields
            result = {}
            for key in ("title", "description", "tags", "categories", "slug",
                        "tldr", "keyTakeaways", "faq", "pubDate", "lang"):
                if key in meta:
                    val = meta[key]
                    if isinstance(val, (list, dict, str, int, float, bool)):
                        result[key] = val
            return result if result else None
        except Exception:
            return None

    def _strip_frontmatter(self, content: str) -> str:
        try:
            post = frontmatter.loads(content)
            return post.content
        except Exception:
            return content

    def _split_by_headings(self, lines: list[str]) -> list[dict]:
        sections: list[dict] = []
        current: dict | None = None

        for i, line in enumerate(lines):
            match = _HEADING_RE.match(line)
            if match:
                if current is not None:
                    current["end_line"] = i  # exclusive
                    current["text"] = "\n".join(lines[current["start_line"] : i])
                    sections.append(current)
                current = {
                    "heading": match.group(2).strip(),
                    "level": len(match.group(1)),
                    "start_line": i + 1,  # 1-indexed
                    "end_line": 0,
                    "text": "",
                }
            elif current is None and line.strip():
                # Content before first heading
                current = {
                    "heading": None,
                    "level": 0,
                    "start_line": 1,
                    "end_line": 0,
                    "text": "",
                }

        # Close last section
        if current is not None:
            current["end_line"] = len(lines)
            current["text"] = "\n".join(lines[current["start_line"] - 1 :])
            sections.append(current)

        return sections

    def _merge_small(self, sections: list[dict]) -> list[dict]:
        if not sections:
            return sections

        merged: list[dict] = [sections[0]]
        for section in sections[1:]:
            if len(section["text"].strip()) < self.min_chunk_chars:
                # Merge into previous
                prev = merged[-1]
                prev["text"] = prev["text"] + "\n\n" + section["text"]
                prev["end_line"] = section["end_line"]
            else:
                merged.append(section)

        return merged

    def _apply_overlap(self, sections: list[dict], all_lines: list[str]) -> list[dict]:
        for i in range(1, len(sections)):
            prev = sections[i - 1]
            prev_end = prev["end_line"]
            overlap_start = max(prev_end - self.overlap_lines, prev["start_line"])
            overlap_text = "\n".join(all_lines[overlap_start - 1 : prev_end])
            sections[i]["text"] = overlap_text + "\n\n" + sections[i]["text"]
            sections[i]["start_line"] = overlap_start
        return sections
```

**Step 5: Update chunker/__init__.py**

```python
"""Chunker modules."""

from dnomia_knowledge.chunker.base import Chunker
from dnomia_knowledge.chunker.md_chunker import MdChunker

__all__ = ["Chunker", "MdChunker"]
```

**Step 6: Run tests**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && python -m pytest tests/test_chunker.py -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/dnomia_knowledge/chunker/ tests/test_chunker.py
git commit -m "feat: heading-based markdown chunker with frontmatter, small chunk merge, and overlap"
```

---

### Task 4: search.py — Hybrid FTS5 + Vector + RRF

**Files:**
- Create: `src/dnomia_knowledge/search.py`
- Create: `tests/test_search.py`

**Step 1: Write the failing tests**

```python
"""Tests for hybrid search."""

from __future__ import annotations

import json

import pytest

from dnomia_knowledge.embedder import Embedder
from dnomia_knowledge.search import HybridSearch
from dnomia_knowledge.store import Store


@pytest.fixture
def populated_store(db_path):
    """Store with chunks and vectors for search testing."""
    store = Store(db_path)
    embedder = Embedder()
    store.register_project("test", "/tmp/test", "content")

    chunks_data = [
        {
            "file_path": "auth.md",
            "chunk_domain": "content",
            "chunk_type": "heading",
            "name": "Authentication",
            "language": "md",
            "content": "JWT token authentication with refresh tokens and session management.",
        },
        {
            "file_path": "database.md",
            "chunk_domain": "content",
            "chunk_type": "heading",
            "name": "Database",
            "language": "md",
            "content": "PostgreSQL database setup with connection pooling and migrations.",
        },
        {
            "file_path": "testing.md",
            "chunk_domain": "content",
            "chunk_type": "heading",
            "name": "Testing",
            "language": "md",
            "content": "Python pytest testing with fixtures, parametrize, and coverage.",
        },
    ]

    chunk_ids = store.insert_chunks("test", chunks_data)
    texts = [c["content"] for c in chunks_data]
    vectors = embedder.embed_passages(texts)
    store.insert_chunk_vectors(chunk_ids, vectors)

    return store, embedder


class TestHybridSearch:
    def test_vector_search(self, populated_store):
        store, embedder = populated_store
        search = HybridSearch(store, embedder)
        results = search.search("JWT authentication", project_id="test")
        assert len(results) > 0
        assert results[0].name == "Authentication"

    def test_fts_search(self, populated_store):
        store, embedder = populated_store
        search = HybridSearch(store, embedder)
        results = search.search("PostgreSQL migrations", project_id="test")
        assert len(results) > 0
        # Database chunk should be relevant
        names = [r.name for r in results]
        assert "Database" in names

    def test_hybrid_combines_results(self, populated_store):
        store, embedder = populated_store
        search = HybridSearch(store, embedder)
        results = search.search("pytest fixtures", project_id="test")
        assert len(results) > 0

    def test_domain_filter(self, populated_store):
        store, embedder = populated_store
        search = HybridSearch(store, embedder)
        results = search.search("authentication", project_id="test", domain="code")
        # No code chunks in test data, should return empty
        assert len(results) == 0

    def test_limit(self, populated_store):
        store, embedder = populated_store
        search = HybridSearch(store, embedder)
        results = search.search("python", project_id="test", limit=1)
        assert len(results) <= 1

    def test_empty_query(self, populated_store):
        store, embedder = populated_store
        search = HybridSearch(store, embedder)
        results = search.search("", project_id="test")
        assert results == []


class TestRRFMerge:
    def test_rrf_basic(self):
        from dnomia_knowledge.search import rrf_merge
        from dnomia_knowledge.models import SearchResult

        list_a = [
            SearchResult(chunk_id=1, project_id="t", file_path="a", chunk_domain="c", chunk_type="h", score=0.9),
            SearchResult(chunk_id=2, project_id="t", file_path="b", chunk_domain="c", chunk_type="h", score=0.8),
        ]
        list_b = [
            SearchResult(chunk_id=2, project_id="t", file_path="b", chunk_domain="c", chunk_type="h", score=0.7),
            SearchResult(chunk_id=3, project_id="t", file_path="c", chunk_domain="c", chunk_type="h", score=0.6),
        ]
        merged = rrf_merge(list_a, list_b, k=60)
        # chunk_id=2 appears in both lists, should rank highest
        assert merged[0].chunk_id == 2

    def test_rrf_respects_limit(self):
        from dnomia_knowledge.search import rrf_merge
        from dnomia_knowledge.models import SearchResult

        list_a = [
            SearchResult(chunk_id=i, project_id="t", file_path=f"{i}", chunk_domain="c", chunk_type="h")
            for i in range(10)
        ]
        merged = rrf_merge(list_a, [], k=60, limit=3)
        assert len(merged) == 3
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && python -m pytest tests/test_search.py -v`
Expected: FAIL

**Step 3: Write search.py**

```python
"""Hybrid search: FTS5 + sqlite-vec + RRF merge."""

from __future__ import annotations

import json
import re

from dnomia_knowledge.embedder import Embedder
from dnomia_knowledge.models import SearchResult
from dnomia_knowledge.store import Store


def rrf_merge(
    list_a: list[SearchResult],
    list_b: list[SearchResult],
    k: int = 60,
    limit: int = 10,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion merge of two ranked lists."""
    scores: dict[int, float] = {}
    result_map: dict[int, SearchResult] = {}

    for rank, r in enumerate(list_a):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1.0 / (k + rank + 1)
        result_map[r.chunk_id] = r

    for rank, r in enumerate(list_b):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1.0 / (k + rank + 1)
        if r.chunk_id not in result_map:
            result_map[r.chunk_id] = r

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:limit]

    results = []
    for chunk_id in sorted_ids:
        r = result_map[chunk_id]
        r.score = round(scores[chunk_id], 6)
        results.append(r)
    return results


def _sanitize_fts_query(query: str) -> str:
    """Remove FTS5 special characters to prevent syntax errors."""
    return re.sub(r'[^\w\s]', ' ', query).strip()


class HybridSearch:
    """Hybrid FTS5 + vector search with RRF merge."""

    def __init__(self, store: Store, embedder: Embedder):
        self._store = store
        self._embedder = embedder

    def search(
        self,
        query: str,
        project_id: str | None = None,
        domain: str = "all",
        limit: int = 10,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []

        # Run FTS5 and vector search in sequence (same thread, same DB)
        fetch_limit = limit * 3  # over-fetch for RRF merge quality
        fts_results = self._search_fts(query, project_id, domain, fetch_limit)
        vec_results = self._search_vector(query, project_id, domain, fetch_limit)

        if not fts_results and not vec_results:
            # Fallback: prefix match
            fts_results = self._search_fts_prefix(query, project_id, domain, fetch_limit)

        return rrf_merge(fts_results, vec_results, k=60, limit=limit)

    def _search_fts(
        self, query: str, project_id: str | None, domain: str, limit: int
    ) -> list[SearchResult]:
        sanitized = _sanitize_fts_query(query)
        if not sanitized:
            return []

        conn = self._store._connect()
        where_clauses = []
        params: list = []

        if project_id:
            where_clauses.append("c.project_id = ?")
            params.append(project_id)
        if domain != "all":
            where_clauses.append("c.chunk_domain = ?")
            params.append(domain)

        where_sql = f"AND {' AND '.join(where_clauses)}" if where_clauses else ""

        sql = f"""
            SELECT c.id, c.project_id, c.file_path, c.chunk_domain, c.chunk_type,
                   c.name, c.language, c.start_line, c.end_line, c.content,
                   bm25(chunks_fts) AS score
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE chunks_fts MATCH ?
            {where_sql}
            ORDER BY score
            LIMIT ?
        """
        params = [sanitized] + params + [limit]

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            return []

        return [self._row_to_result(r) for r in rows]

    def _search_fts_prefix(
        self, query: str, project_id: str | None, domain: str, limit: int
    ) -> list[SearchResult]:
        """Layer 2 fallback: prefix match."""
        sanitized = _sanitize_fts_query(query)
        if not sanitized:
            return []
        prefix_query = " ".join(f"{word}*" for word in sanitized.split())
        conn = self._store._connect()

        where_clauses = []
        params: list = []
        if project_id:
            where_clauses.append("c.project_id = ?")
            params.append(project_id)
        if domain != "all":
            where_clauses.append("c.chunk_domain = ?")
            params.append(domain)
        where_sql = f"AND {' AND '.join(where_clauses)}" if where_clauses else ""

        sql = f"""
            SELECT c.id, c.project_id, c.file_path, c.chunk_domain, c.chunk_type,
                   c.name, c.language, c.start_line, c.end_line, c.content,
                   bm25(chunks_fts) AS score
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE chunks_fts MATCH ?
            {where_sql}
            ORDER BY score
            LIMIT ?
        """
        params = [prefix_query] + params + [limit]
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            return []
        return [self._row_to_result(r) for r in rows]

    def _search_vector(
        self, query: str, project_id: str | None, domain: str, limit: int
    ) -> list[SearchResult]:
        query_vec = self._embedder.embed_query(query)
        conn = self._store._connect()

        # sqlite-vec KNN search
        vec_sql = """
            SELECT id, distance
            FROM chunks_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        """
        try:
            vec_rows = conn.execute(vec_sql, (json.dumps(query_vec), limit)).fetchall()
        except Exception:
            return []

        if not vec_rows:
            return []

        # Get chunk details for matched IDs
        chunk_ids = [r[0] for r in vec_rows]
        distances = {r[0]: r[1] for r in vec_rows}
        placeholders = ",".join("?" * len(chunk_ids))

        where_clauses = [f"c.id IN ({placeholders})"]
        params: list = list(chunk_ids)

        if project_id:
            where_clauses.append("c.project_id = ?")
            params.append(project_id)
        if domain != "all":
            where_clauses.append("c.chunk_domain = ?")
            params.append(domain)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT c.id, c.project_id, c.file_path, c.chunk_domain, c.chunk_type,
                   c.name, c.language, c.start_line, c.end_line, c.content
            FROM chunks c
            WHERE {where_sql}
        """

        rows = conn.execute(sql, params).fetchall()

        results = []
        for r in rows:
            result = self._row_to_result(r)
            dist = distances.get(r[0], 1.0)
            result.score = round(1.0 / (1.0 + dist), 6)  # distance to similarity
            results.append(result)

        # Sort by score descending (highest similarity first)
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _row_to_result(self, row) -> SearchResult:
        content = row["content"] if "content" in row.keys() else ""
        snippet_lines = content.split("\n")[:10]
        snippet = "\n".join(snippet_lines)
        if len(content.split("\n")) > 10:
            snippet += "\n..."

        return SearchResult(
            chunk_id=row["id"] if "id" in row.keys() else row[0],
            project_id=row["project_id"] if "project_id" in row.keys() else "",
            file_path=row["file_path"] if "file_path" in row.keys() else "",
            chunk_domain=row["chunk_domain"] if "chunk_domain" in row.keys() else "",
            chunk_type=row["chunk_type"] if "chunk_type" in row.keys() else "",
            name=row["name"] if "name" in row.keys() else None,
            language=row["language"] if "language" in row.keys() else None,
            start_line=row["start_line"] if "start_line" in row.keys() else 0,
            end_line=row["end_line"] if "end_line" in row.keys() else 0,
            score=row["score"] if "score" in row.keys() else 0.0,
            snippet=snippet,
        )
```

**Step 4: Run tests**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && python -m pytest tests/test_search.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/dnomia_knowledge/search.py tests/test_search.py
git commit -m "feat: hybrid search with FTS5 + sqlite-vec + RRF merge and prefix fallback"
```

---

### Task 5: indexer.py — Markdown Indexing Pipeline

**Files:**
- Create: `src/dnomia_knowledge/indexer.py`
- Create: `tests/test_indexer.py`

**Step 1: Write the failing tests**

```python
"""Tests for indexing pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from dnomia_knowledge.embedder import Embedder
from dnomia_knowledge.indexer import Indexer
from dnomia_knowledge.store import Store


@pytest.fixture
def indexer(db_path):
    store = Store(db_path)
    embedder = Embedder()
    return Indexer(store, embedder)


class TestIndexer:
    def test_index_single_file(self, indexer, sample_markdown_file):
        result = indexer.index_file(
            project_id="test",
            project_path=str(sample_markdown_file.parent),
            file_path=str(sample_markdown_file),
        )
        assert result > 0  # at least 1 chunk

    def test_index_directory(self, indexer, tmp_dir, sample_markdown):
        # Create multiple markdown files
        (tmp_dir / "post1.md").write_text(sample_markdown)
        (tmp_dir / "post2.md").write_text("## Another\n\nAnother post about databases.")
        (tmp_dir / "ignored.pyc").write_bytes(b"\x00\x01\x02")

        result = indexer.index_directory("test-proj", str(tmp_dir))
        assert result.total_files == 2  # .pyc skipped
        assert result.total_chunks > 0

    def test_incremental_reindex(self, indexer, tmp_dir, sample_markdown):
        md_file = tmp_dir / "post.md"
        md_file.write_text(sample_markdown)

        # First index
        r1 = indexer.index_directory("test-proj", str(tmp_dir))
        first_count = r1.total_chunks

        # Second index without changes — should be no-op
        r2 = indexer.index_directory("test-proj", str(tmp_dir))
        assert r2.indexed_files == 0

        # Modify file
        md_file.write_text(sample_markdown + "\n## New Section\n\nNew content here.")
        r3 = indexer.index_directory("test-proj", str(tmp_dir))
        assert r3.indexed_files == 1

    def test_deleted_file_cleanup(self, indexer, tmp_dir, sample_markdown):
        md_file = tmp_dir / "post.md"
        md_file.write_text(sample_markdown)

        indexer.index_directory("test-proj", str(tmp_dir))
        md_file.unlink()
        result = indexer.index_directory("test-proj", str(tmp_dir))
        stats = indexer.store.get_project_stats("test-proj")
        assert stats["total_chunks"] == 0

    def test_binary_files_skipped(self, indexer, tmp_dir):
        (tmp_dir / "image.png").write_bytes(b"\x89PNG\r\n")
        (tmp_dir / "readme.md").write_text("## Hello\n\nWorld")
        result = indexer.index_directory("test-proj", str(tmp_dir))
        assert result.total_files == 1

    def test_gitignore_respected(self, indexer, tmp_dir):
        (tmp_dir / ".gitignore").write_text("ignored/\n")
        (tmp_dir / "ignored").mkdir()
        (tmp_dir / "ignored" / "secret.md").write_text("## Secret\n\nDon't index this")
        (tmp_dir / "public.md").write_text("## Public\n\nIndex this")
        result = indexer.index_directory("test-proj", str(tmp_dir))
        assert result.total_files == 1
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && python -m pytest tests/test_indexer.py -v`
Expected: FAIL

**Step 3: Write indexer.py**

Kaynak: mcp-code-search/indexer.py'den file scanning, gitignore, binary skip, incremental reindex mantigi alinacak.

```python
"""Indexing pipeline: scan -> chunk -> embed -> store."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

import pathspec

from dnomia_knowledge.chunker.md_chunker import MdChunker
from dnomia_knowledge.embedder import Embedder
from dnomia_knowledge.models import IndexResult
from dnomia_knowledge.store import Store

logger = logging.getLogger(__name__)

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a",
    ".pyc", ".pyo", ".class", ".jar",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".db", ".sqlite", ".sqlite3",
    ".DS_Store",
}

DEFAULT_IGNORE_PATTERNS = [
    "node_modules", "dist", ".next", "__pycache__", ".git",
    "*.egg-info", ".venv", "venv", ".tox", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", "build",
]

MAX_FILE_SIZE_KB = 500


class Indexer:
    """Scan, chunk, embed, and store pipeline."""

    def __init__(self, store: Store, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self._md_chunker = MdChunker()

    def index_file(
        self,
        project_id: str,
        project_path: str,
        file_path: str,
    ) -> int:
        """Index a single file. Returns number of chunks created."""
        content = self._read_file(file_path)
        if not content:
            return 0

        rel_path = os.path.relpath(file_path, project_path)
        chunks = self._md_chunker.chunk(file_path, content)
        if not chunks:
            return 0

        # Delete old chunks for this file
        self.store.delete_file_chunks(project_id, rel_path)

        # Prepare chunk data
        chunk_dicts = [
            {
                "file_path": rel_path,
                "chunk_domain": "content",
                "chunk_type": c.chunk_type,
                "name": c.name,
                "language": c.language,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "content": c.content,
                "metadata": _serialize_metadata(c.metadata),
            }
            for c in chunks
        ]

        # Insert chunks
        chunk_ids = self.store.insert_chunks(project_id, chunk_dicts)

        # Embed and store vectors
        texts = [c.content for c in chunks]
        vectors = self.embedder.embed_passages(texts)
        self.store.insert_chunk_vectors(chunk_ids, vectors)

        # Update file index
        file_hash = _compute_file_hash(file_path)
        self.store.upsert_file_index(project_id, rel_path, file_hash, len(chunk_ids))

        return len(chunk_ids)

    def index_directory(
        self,
        project_id: str,
        directory: str,
        incremental: bool = True,
        ignore_patterns: list[str] | None = None,
    ) -> IndexResult:
        """Index all markdown files in a directory."""
        start_time = time.time()
        project_path = str(Path(directory).resolve())

        # Register project
        self.store.register_project(project_id, project_path, "content")

        # Set embedding metadata on first use
        if not self.store.get_metadata("embedding_model"):
            self.store.set_metadata("embedding_model", self.embedder.model_name)
            self.store.set_metadata("embedding_dim", str(self.embedder.dimension))

        # Scan files
        patterns = ignore_patterns or DEFAULT_IGNORE_PATTERNS
        all_files = self._scan_files(project_path, patterns)

        if incremental:
            stored_hashes = self.store.get_all_file_hashes(project_id)
            changed, deleted = self._detect_changes(all_files, stored_hashes, project_path)

            # Clean up deleted files
            for rel_path in deleted:
                self.store.delete_file_chunks(project_id, rel_path)
                self.store.delete_file_index(project_id, rel_path)

            files_to_index = changed
        else:
            files_to_index = all_files

        # Index changed files
        total_chunks = 0
        for file_path in files_to_index:
            try:
                count = self.index_file(project_id, project_path, file_path)
                total_chunks += count
            except Exception as e:
                logger.warning("Failed to index %s: %s", file_path, e)

        duration = time.time() - start_time
        return IndexResult(
            project_id=project_id,
            total_files=len(all_files),
            indexed_files=len(files_to_index),
            total_chunks=total_chunks,
            content_chunks=total_chunks,
            duration_seconds=round(duration, 2),
        )

    def _scan_files(self, project_path: str, ignore_patterns: list[str]) -> list[str]:
        """Scan for markdown files, respecting .gitignore and ignore patterns."""
        root = Path(project_path)

        # Load .gitignore
        gitignore_spec = None
        gitignore_path = root / ".gitignore"
        if gitignore_path.exists():
            try:
                lines = gitignore_path.read_text().splitlines()
                gitignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
            except Exception:
                pass

        config_spec = pathspec.PathSpec.from_lines("gitwildmatch", ignore_patterns)
        max_size = MAX_FILE_SIZE_KB * 1024
        files = []

        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root)

            # Filter directories in-place
            dirnames[:] = [
                d for d in dirnames
                if not self._should_ignore_dir(d, rel_dir, config_spec, gitignore_spec)
            ]

            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root)
                ext = os.path.splitext(fname)[1].lower()

                # Skip binary
                if ext in BINARY_EXTENSIONS:
                    continue

                # Sprint 1: only markdown files
                if ext not in (".md", ".mdx"):
                    continue

                # Skip ignored
                if config_spec.match_file(rel_path):
                    continue
                if gitignore_spec and gitignore_spec.match_file(rel_path):
                    continue

                # Skip large files
                try:
                    if os.path.getsize(full_path) > max_size:
                        continue
                except OSError:
                    continue

                files.append(full_path)

        return files

    def _should_ignore_dir(
        self,
        dirname: str,
        parent_rel: str,
        config_spec: pathspec.PathSpec,
        gitignore_spec: pathspec.PathSpec | None,
    ) -> bool:
        if dirname.startswith(".") and dirname not in (".github", ".vscode", ".config"):
            return True

        rel_path = os.path.join(parent_rel, dirname) if parent_rel != "." else dirname

        if config_spec.match_file(rel_path + "/"):
            return True
        if config_spec.match_file(dirname):
            return True

        if gitignore_spec:
            if gitignore_spec.match_file(rel_path + "/"):
                return True
            if gitignore_spec.match_file(dirname + "/"):
                return True

        return False

    def _detect_changes(
        self,
        all_files: list[str],
        stored_hashes: dict[str, str],
        project_path: str,
    ) -> tuple[list[str], list[str]]:
        """Returns (changed_files, deleted_rel_paths)."""
        current_rel_paths = set()
        changed = []

        for fp in all_files:
            rel = os.path.relpath(fp, project_path)
            current_rel_paths.add(rel)
            current_hash = _compute_file_hash(fp)
            if rel not in stored_hashes or stored_hashes[rel] != current_hash:
                changed.append(fp)

        deleted = [rp for rp in stored_hashes if rp not in current_rel_paths]
        return changed, deleted

    def _read_file(self, file_path: str) -> str | None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return None


def _compute_file_hash(file_path: str) -> str:
    """MD5 content hash."""
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


def _serialize_metadata(meta: dict | None) -> str | None:
    if meta is None:
        return None
    import json
    return json.dumps(meta, ensure_ascii=False, default=str)
```

**Step 4: Run tests**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && python -m pytest tests/test_indexer.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/dnomia_knowledge/indexer.py tests/test_indexer.py
git commit -m "feat: indexing pipeline with scan, chunk, embed, store, incremental reindex"
```

---

### Task 6: server.py — MCP Server with search + index_project

**Files:**
- Create: `src/dnomia_knowledge/server.py`
- Create: `src/dnomia_knowledge/__main__.py`
- Create: `tests/test_server.py`

**Step 1: Write the failing tests**

```python
"""Tests for MCP server tool definitions."""

from __future__ import annotations

import pytest

from dnomia_knowledge.server import create_server


class TestServerCreation:
    def test_server_created(self):
        server = create_server()
        assert server is not None
        assert server.name == "dnomia-knowledge"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && python -m pytest tests/test_server.py -v`
Expected: FAIL

**Step 3: Write server.py**

```python
"""FastMCP server for dnomia-knowledge."""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from dnomia_knowledge.embedder import Embedder
from dnomia_knowledge.indexer import Indexer
from dnomia_knowledge.search import HybridSearch
from dnomia_knowledge.store import Store

# Default DB path
_DEFAULT_DB_PATH = os.path.expanduser("~/.local/share/dnomia-knowledge/knowledge.db")

# Lazy-loaded singletons
_store: Store | None = None
_embedder: Embedder | None = None
_search: HybridSearch | None = None
_indexer: Indexer | None = None


def _get_store() -> Store:
    global _store
    if _store is None:
        db_path = os.environ.get("DNOMIA_KNOWLEDGE_DB", _DEFAULT_DB_PATH)
        _store = Store(db_path)
    return _store


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_search() -> HybridSearch:
    global _search
    if _search is None:
        _search = HybridSearch(_get_store(), _get_embedder())
    return _search


def _get_indexer() -> Indexer:
    global _indexer
    if _indexer is None:
        _indexer = Indexer(_get_store(), _get_embedder())
    return _indexer


def _default_project() -> str | None:
    return os.environ.get("DNOMIA_KNOWLEDGE_PROJECT")


def create_server() -> FastMCP:
    server = FastMCP("dnomia-knowledge")

    @server.tool()
    async def search(
        query: str,
        domain: str = "all",
        project: str | None = None,
        cross: bool = False,
        limit: int = 10,
    ) -> str:
        """Hybrid semantic + keyword search across project knowledge.

        Args:
            query: Search query text
            domain: Filter by "all", "code", or "content"
            project: Project ID (default: DNOMIA_KNOWLEDGE_PROJECT env var)
            cross: If True, also search linked projects
            limit: Maximum results to return
        """
        if not query or not query.strip():
            return "Empty query."

        project_id = project or _default_project()
        searcher = _get_search()
        results = searcher.search(query, project_id=project_id, domain=domain, limit=limit)

        if not results:
            return "No results found."

        lines = []
        for i, r in enumerate(results, 1):
            header = f"[{i}] {r.file_path}"
            if r.start_line:
                header += f":{r.start_line}-{r.end_line}"
            header += f" (score: {r.score:.4f}"
            if r.project_id:
                header += f", project: {r.project_id}"
            header += ")"

            name_line = ""
            if r.name:
                name_line = f"    {r.chunk_type} {r.name}"
            elif r.chunk_type:
                name_line = f"    [{r.chunk_type}]"

            snippet = ""
            if r.snippet:
                snippet_lines = r.snippet.split("\n")[:5]
                snippet = "\n".join(f"    {l}" for l in snippet_lines)

            parts = [header]
            if name_line:
                parts.append(name_line)
            if snippet:
                parts.append(snippet)
            lines.append("\n".join(parts))

        return "\n\n".join(lines)

    @server.tool()
    async def index_project(
        path: str,
        incremental: bool = True,
    ) -> str:
        """Index or reindex a project's content and code.

        Args:
            path: Absolute path to project root (must contain .md/.mdx files)
            incremental: If True, only reindex changed files (default)
        """
        if not Path(path).is_dir():
            return f"Error: {path} is not a directory."

        indexer = _get_indexer()

        # Derive project ID from directory name
        project_id = Path(path).name.lower().replace(" ", "-")

        result = indexer.index_directory(
            project_id=project_id,
            directory=path,
            incremental=incremental,
        )

        return (
            f"Indexed {result.project_id}: "
            f"{result.total_files} files, "
            f"{result.total_chunks} chunks "
            f"({result.content_chunks} content + {result.code_chunks} code)\n"
            f"  Changed: {result.indexed_files} files\n"
            f"  Duration: {result.duration_seconds}s"
        )

    @server.tool()
    async def project_info(
        project: str | None = None,
    ) -> str:
        """List registered projects with stats.

        Args:
            project: Specific project ID, or None for all projects
        """
        store = _get_store()

        if project:
            proj = store.get_project(project)
            if not proj:
                return f"Project '{project}' not found."
            stats = store.get_project_stats(project)
            return (
                f"{proj['id']} ({proj['type']})\n"
                f"  Path: {proj['path']}\n"
                f"  Chunks: {stats['total_chunks']} "
                f"({stats['content_chunks']} content, {stats['code_chunks']} code)\n"
                f"  Files: {stats['total_files']}\n"
                f"  Graph: {'enabled' if proj['graph_enabled'] else 'disabled'}\n"
                f"  Last indexed: {proj['last_indexed'] or 'never'}"
            )

        projects = store.list_projects()
        if not projects:
            return "No projects registered. Use index_project to add one."

        lines = []
        for p in projects:
            stats = store.get_project_stats(p["id"])
            lines.append(
                f"- {p['id']} ({p['type']}): "
                f"{stats['total_chunks']} chunks, "
                f"{stats['total_files']} files"
            )
        return "\n".join(lines)

    return server


# Entry point
server = create_server()

if __name__ == "__main__":
    server.run()
```

**Step 4: Write __main__.py**

```python
"""Allow running as `python -m dnomia_knowledge`."""

from dnomia_knowledge.server import server

server.run()
```

**Step 5: Run tests**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && python -m pytest tests/test_server.py -v`
Expected: ALL PASS

**Step 6: Smoke test — end-to-end**

```bash
cd /Users/ceair/Desktop/dnomia-knowledge
source .venv/bin/activate

# Create a test directory with markdown
mkdir -p /tmp/test-knowledge
cat > /tmp/test-knowledge/auth.md << 'EOF'
---
title: Authentication Guide
tags: [auth, jwt, security]
---

## JWT Tokens

JSON Web Tokens are used for stateless authentication. They contain encoded claims.

## Refresh Tokens

Refresh tokens allow obtaining new access tokens without re-authentication.

## Session Management

Server-side sessions can complement JWT for enhanced security.
EOF

# Test via Python REPL
python3 -c "
from dnomia_knowledge.store import Store
from dnomia_knowledge.embedder import Embedder
from dnomia_knowledge.indexer import Indexer
from dnomia_knowledge.search import HybridSearch

store = Store('/tmp/test-knowledge/test.db')
embedder = Embedder()
indexer = Indexer(store, embedder)

# Index
result = indexer.index_directory('test', '/tmp/test-knowledge')
print(f'Indexed: {result.total_chunks} chunks from {result.total_files} files')

# Search
search = HybridSearch(store, embedder)
results = search.search('JWT authentication', project_id='test')
print(f'Found {len(results)} results')
for r in results:
    print(f'  [{r.score:.4f}] {r.name}: {r.snippet[:60]}...')

store.close()
print('OK')
"
```

Expected: Chunks indexed, search returns relevant results with JWT/auth chunk ranked highest.

**Step 7: Commit**

```bash
git add src/dnomia_knowledge/server.py src/dnomia_knowledge/__main__.py tests/test_server.py
git commit -m "feat: MCP server with search and index_project tools"
```

---

### Task 7: Run All Tests + Final Verification

**Step 1: Run full test suite**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 2: Lint**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && ruff check src/ tests/`
Expected: No errors (or only minor formatting)

**Step 3: Fix any lint issues**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && ruff check src/ tests/ --fix`

**Step 4: Final commit if any fixes**

```bash
git add -A
git commit -m "fix: lint and formatting cleanup"
```

**Step 5: Verify MCP server starts**

Run: `cd /Users/ceair/Desktop/dnomia-knowledge && source .venv/bin/activate && timeout 5 python -m dnomia_knowledge.server 2>&1 || true`
Expected: Server starts without errors (will timeout after 5s since it waits for stdio)

---

## Sprint 1 Summary

After completing all tasks:

```
dnomia-knowledge/
├── src/dnomia_knowledge/
│   ├── __init__.py
│   ├── __main__.py          ← python -m dnomia_knowledge
│   ├── server.py            ← FastMCP, 3 tools (search, index_project, project_info)
│   ├── store.py             ← SQLite + FTS5 + sqlite-vec, 9 tables
│   ├── embedder.py          ← multilingual-e5-base, query/passage prefix, lazy load
│   ├── search.py            ← Hybrid FTS5 + vec + RRF
│   ├── indexer.py           ← Scan → chunk → embed → store pipeline
│   ├── models.py            ← Pydantic models
│   └── chunker/
│       ├── __init__.py
│       ├── base.py          ← Chunker protocol
│       └── md_chunker.py    ← Heading-based markdown chunking
├── tests/
│   ├── conftest.py
│   ├── test_store.py
│   ├── test_embedder.py
│   ├── test_chunker.py
│   ├── test_search.py
│   ├── test_indexer.py
│   └── test_server.py
├── pyproject.toml
├── .gitignore
└── .python-version
```

**Capabilities at end of Sprint 1:**
- Markdown dosyalarini heading-based chunking ile indeksle
- Hybrid search: FTS5 keyword + vector semantic + RRF merge
- Incremental reindex (MD5 content hash)
- MCP server olarak Claude Code'dan kullanilabilir
- CLI yok (Sprint 2), AST chunking yok (Sprint 3), graph yok (Sprint 3)
