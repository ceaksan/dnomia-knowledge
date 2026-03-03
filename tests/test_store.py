"""Tests for SQLite store."""

from __future__ import annotations

import json

from dnomia_knowledge.store import Store


class TestStoreInit:
    def test_creates_database(self, db_path):
        store = Store(db_path)
        assert store.db_path == db_path

    def test_creates_tables(self, db_path):
        store = Store(db_path)
        conn = store._connect()
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
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
