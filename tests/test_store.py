"""Tests for SQLite store."""

from __future__ import annotations

import json

from dnomia_knowledge.models import InteractionType
from dnomia_knowledge.store import Store


def test_interaction_type_enum_values():
    assert InteractionType.READ == "read"
    assert InteractionType.EDIT == "edit"
    assert InteractionType.SEARCH_HIT == "search_hit"


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
        assert row[0] == "2"


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


def _setup_project_with_chunks(db_path: str, n_chunks: int = 3) -> tuple:
    """Helper: create project with n chunks, return (store, chunk_ids)."""
    store = Store(db_path)
    store.register_project("test", "/tmp/test", "content")
    chunks = [
        {"file_path": f"file{i}.md", "content": f"chunk {i} content"} for i in range(n_chunks)
    ]
    ids = store.insert_chunks("test", chunks)
    return store, ids


class TestEdgeCRUD:
    def test_insert_edges(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        count = store.insert_edges(
            [
                {"source_id": ids[0], "target_id": ids[1], "edge_type": "link", "weight": 0.9},
                {"source_id": ids[1], "target_id": ids[2], "edge_type": "tag"},
            ]
        )
        assert count == 2

    def test_insert_duplicate_ignored(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        store.insert_edges(
            [
                {"source_id": ids[0], "target_id": ids[1], "edge_type": "link"},
            ]
        )
        count = store.insert_edges(
            [
                {"source_id": ids[0], "target_id": ids[1], "edge_type": "link"},
            ]
        )
        assert count == 0

    def test_delete_edges_for_chunk(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        store.insert_edges(
            [
                {"source_id": ids[0], "target_id": ids[1], "edge_type": "link"},
                {"source_id": ids[2], "target_id": ids[0], "edge_type": "tag"},
                {"source_id": ids[1], "target_id": ids[2], "edge_type": "link"},
            ]
        )
        deleted = store.delete_edges_for_chunk(ids[0])
        assert deleted == 2  # one as source, one as target

    def test_delete_edges_for_chunk_by_type(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        store.insert_edges(
            [
                {"source_id": ids[0], "target_id": ids[1], "edge_type": "link"},
                {"source_id": ids[0], "target_id": ids[2], "edge_type": "tag"},
            ]
        )
        deleted = store.delete_edges_for_chunk(ids[0], edge_type="link")
        assert deleted == 1
        # tag edge should remain
        edges = store.get_edges_for_project("test")
        assert len(edges) == 1
        assert edges[0]["edge_type"] == "tag"

    def test_delete_edges_for_project(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        store.insert_edges(
            [
                {"source_id": ids[0], "target_id": ids[1], "edge_type": "link"},
                {"source_id": ids[1], "target_id": ids[2], "edge_type": "semantic"},
            ]
        )
        deleted = store.delete_edges_for_project("test")
        assert deleted == 2

    def test_delete_edges_for_project_by_type(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        store.insert_edges(
            [
                {"source_id": ids[0], "target_id": ids[1], "edge_type": "link"},
                {"source_id": ids[1], "target_id": ids[2], "edge_type": "semantic"},
            ]
        )
        deleted = store.delete_edges_for_project("test", edge_type="link")
        assert deleted == 1

    def test_get_neighbors_depth1(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        store.insert_edges(
            [
                {"source_id": ids[0], "target_id": ids[1], "edge_type": "link"},
                {"source_id": ids[1], "target_id": ids[2], "edge_type": "link"},
            ]
        )
        neighbors = store.get_neighbors(ids[0], depth=1)
        neighbor_ids = {n["chunk_id"] for n in neighbors}
        assert ids[1] in neighbor_ids
        assert ids[2] not in neighbor_ids

    def test_get_neighbors_depth2(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        store.insert_edges(
            [
                {"source_id": ids[0], "target_id": ids[1], "edge_type": "link"},
                {"source_id": ids[1], "target_id": ids[2], "edge_type": "link"},
            ]
        )
        neighbors = store.get_neighbors(ids[0], depth=2)
        neighbor_ids = {n["chunk_id"] for n in neighbors}
        assert ids[1] in neighbor_ids
        assert ids[2] in neighbor_ids

    def test_get_neighbors_with_edge_type_filter(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        store.insert_edges(
            [
                {"source_id": ids[0], "target_id": ids[1], "edge_type": "link"},
                {"source_id": ids[0], "target_id": ids[2], "edge_type": "semantic"},
            ]
        )
        neighbors = store.get_neighbors(ids[0], depth=1, edge_types=["link"])
        assert len(neighbors) == 1
        assert neighbors[0]["edge_type"] == "link"

    def test_get_edges_for_project(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        store.insert_edges(
            [
                {"source_id": ids[0], "target_id": ids[1], "edge_type": "link"},
            ]
        )
        edges = store.get_edges_for_project("test")
        assert len(edges) == 1
        assert edges[0]["source_id"] == ids[0]

    def test_get_project_edge_stats(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        store.insert_edges(
            [
                {"source_id": ids[0], "target_id": ids[1], "edge_type": "link"},
                {"source_id": ids[0], "target_id": ids[2], "edge_type": "link"},
                {"source_id": ids[1], "target_id": ids[2], "edge_type": "tag"},
            ]
        )
        stats = store.get_project_edge_stats("test")
        assert stats["link"] == 2
        assert stats["tag"] == 1

    def test_get_chunk_ids_for_project(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        result = store.get_chunk_ids_for_project("test")
        assert result == sorted(ids)

    def test_update_chunk_metadata_merge(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        chunk_ids = store.insert_chunks(
            "test",
            [{"file_path": "a.md", "content": "test", "metadata": json.dumps({"key1": "val1"})}],
        )
        store.update_chunk_metadata(chunk_ids[0], {"key2": "val2"})
        conn = store._connect()
        row = conn.execute("SELECT metadata FROM chunks WHERE id = ?", (chunk_ids[0],)).fetchone()
        meta = json.loads(row[0])
        assert meta["key1"] == "val1"
        assert meta["key2"] == "val2"

    def test_update_chunk_metadata_overwrite(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        chunk_ids = store.insert_chunks(
            "test",
            [{"file_path": "a.md", "content": "test", "metadata": json.dumps({"key": "old"})}],
        )
        store.update_chunk_metadata(chunk_ids[0], {"key": "new"})
        conn = store._connect()
        row = conn.execute("SELECT metadata FROM chunks WHERE id = ?", (chunk_ids[0],)).fetchone()
        meta = json.loads(row[0])
        assert meta["key"] == "new"

    def test_update_chunk_metadata_null_initial(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        chunk_ids = store.insert_chunks(
            "test",
            [{"file_path": "a.md", "content": "test"}],
        )
        store.update_chunk_metadata(chunk_ids[0], {"community_id": 3})
        conn = store._connect()
        row = conn.execute("SELECT metadata FROM chunks WHERE id = ?", (chunk_ids[0],)).fetchone()
        meta = json.loads(row[0])
        assert meta["community_id"] == 3


class TestInteractionCRUD:
    def test_log_interaction(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        store.log_interaction(ids[0], "read", "Read", project_id="test", file_path="file0.md")
        conn = store._connect()
        row = conn.execute(
            "SELECT * FROM chunk_interactions WHERE chunk_id = ?", (ids[0],)
        ).fetchone()
        assert row is not None
        assert row["interaction"] == "read"
        assert row["source_tool"] == "Read"
        assert row["project_id"] == "test"
        assert row["file_path"] == "file0.md"

    def test_log_interaction_without_file_info(self, db_path):
        """log_interaction should work without project_id/file_path (backward compat)."""
        store, ids = _setup_project_with_chunks(db_path)
        store.log_interaction(ids[0], "read", "Read")
        conn = store._connect()
        row = conn.execute(
            "SELECT * FROM chunk_interactions WHERE chunk_id = ?", (ids[0],)
        ).fetchone()
        assert row is not None
        assert row["project_id"] is None
        assert row["file_path"] is None

    def test_get_interaction_counts(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        store.log_interaction(ids[0], "read", "Read")
        store.log_interaction(ids[0], "read", "Read")
        store.log_interaction(ids[0], "edit", "Edit")
        store.log_interaction(ids[1], "read", "Read")

        counts = store.get_interaction_counts(ids)
        assert counts[ids[0]] == 3
        assert counts[ids[1]] == 1
        assert ids[2] not in counts

    def test_interaction_counts_filter_by_type(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        store.log_interaction(ids[0], "read", "Read")
        store.log_interaction(ids[0], "edit", "Edit")
        store.log_interaction(ids[0], "search_hit", "search")

        counts = store.get_interaction_counts(ids, interactions=["read", "edit"])
        assert counts[ids[0]] == 2

    def test_interaction_counts_filter_by_project(self, db_path):
        store = Store(db_path)
        store.register_project("proj-a", "/tmp/a", "content")
        store.register_project("proj-b", "/tmp/b", "content")
        ids_a = store.insert_chunks("proj-a", [{"file_path": "a.md", "content": "a"}])
        ids_b = store.insert_chunks("proj-b", [{"file_path": "b.md", "content": "b"}])

        store.log_interaction(ids_a[0], "read", "Read")
        store.log_interaction(ids_b[0], "read", "Read")

        counts_a = store.get_interaction_counts(ids_a + ids_b, project_id="proj-a")
        assert ids_a[0] in counts_a
        assert ids_b[0] not in counts_a

    def test_delete_old_interactions(self, db_path):
        store, ids = _setup_project_with_chunks(db_path)
        store.log_interaction(ids[0], "read", "Read")
        # Set timestamp to 100 days ago
        conn = store._connect()
        conn.execute("UPDATE chunk_interactions SET timestamp = datetime('now', '-100 days')")
        conn.commit()
        deleted = store.delete_old_interactions(days=90)
        assert deleted == 1


class TestSearchLogCRUD:
    def test_log_search(self, db_path):
        store = Store(db_path)
        store.log_search("test query", "proj", "all", [1, 2, 3], 3)
        logs = store.get_search_log()
        assert len(logs) == 1
        assert logs[0]["query"] == "test query"
        assert logs[0]["result_count"] == 3

    def test_get_search_log_filter_project(self, db_path):
        store = Store(db_path)
        store.log_search("q1", "proj-a", "all", [1], 1)
        store.log_search("q2", "proj-b", "all", [2], 1)
        logs = store.get_search_log(project_id="proj-a")
        assert len(logs) == 1
        assert logs[0]["query"] == "q1"

    def test_delete_old_search_logs(self, db_path):
        store = Store(db_path)
        store.log_search("old query", None, "all", [], 0)
        conn = store._connect()
        conn.execute("UPDATE search_log SET timestamp = datetime('now', '-100 days')")
        conn.commit()
        deleted = store.delete_old_search_logs(days=90)
        assert deleted == 1

    def test_get_chunk_ids_for_file(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        store.insert_chunks(
            "test",
            [
                {"file_path": "a.md", "content": "chunk 1"},
                {"file_path": "a.md", "content": "chunk 2"},
                {"file_path": "b.md", "content": "chunk 3"},
            ],
        )
        ids = store.get_chunk_ids_for_file("test", "a.md")
        assert len(ids) == 2

        ids_b = store.get_chunk_ids_for_file("test", "b.md")
        assert len(ids_b) == 1


class TestFileChunkQueries:
    def test_get_chunks_for_file(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        store.insert_chunks(
            "test",
            [
                {
                    "file_path": "a.py",
                    "chunk_type": "function",
                    "name": "bar",
                    "language": "python",
                    "start_line": 20,
                    "end_line": 30,
                    "content": "def bar(): pass",
                },
                {
                    "file_path": "a.py",
                    "chunk_type": "function",
                    "name": "foo",
                    "language": "python",
                    "start_line": 1,
                    "end_line": 10,
                    "content": "def foo(): pass",
                },
            ],
        )
        chunks = store.get_chunks_for_file("test", "a.py")
        assert len(chunks) == 2
        # Ordered by start_line
        assert chunks[0]["name"] == "foo"
        assert chunks[0]["start_line"] == 1
        assert chunks[1]["name"] == "bar"
        assert chunks[1]["start_line"] == 20
        # All expected fields present
        assert "id" in chunks[0]
        assert "chunk_type" in chunks[0]
        assert "language" in chunks[0]
        assert "content" in chunks[0]

    def test_get_chunks_for_file_empty(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        chunks = store.get_chunks_for_file("test", "nonexistent.py")
        assert chunks == []

    def test_get_file_line_count(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        store.insert_chunks(
            "test",
            [
                {
                    "file_path": "a.py",
                    "start_line": 1,
                    "end_line": 50,
                    "content": "first chunk",
                },
                {
                    "file_path": "a.py",
                    "start_line": 51,
                    "end_line": 120,
                    "content": "second chunk",
                },
            ],
        )
        line_count = store.get_file_line_count("test", "a.py")
        assert line_count == 120

    def test_get_file_line_count_unknown(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        line_count = store.get_file_line_count("test", "nonexistent.py")
        assert line_count is None


class TestInteractionSurvival:
    def test_interactions_survive_reindex(self, db_path):
        """Interactions should not be deleted when chunks are re-indexed."""
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        ids = store.insert_chunks(
            "test",
            [{"file_path": "a.md", "content": "old content", "name": "Introduction"}],
        )
        store.log_interaction(ids[0], "read", "Read", project_id="test", file_path="a.md")
        store.log_interaction(ids[0], "read", "Read", project_id="test", file_path="a.md")

        # Simulate re-index: delete old chunks, insert new
        store.delete_file_chunks("test", "a.md")
        store.insert_chunks(
            "test",
            [{"file_path": "a.md", "content": "new content", "name": "Introduction"}],
        )

        # Old interactions should still exist (keyed by project_id + file_path)
        conn = store._connect()
        count = conn.execute(
            "SELECT COUNT(*) FROM chunk_interactions WHERE project_id = ? AND file_path = ?",
            ("test", "a.md"),
        ).fetchone()[0]
        assert count == 2

    def test_orphaned_interactions_cleaned_by_gc(self, db_path):
        """gc should remove interactions whose chunk_id no longer exists."""
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        ids = store.insert_chunks(
            "test",
            [{"file_path": "a.md", "content": "content"}],
        )
        store.log_interaction(ids[0], "read", "Read", project_id="test", file_path="a.md")
        store.delete_file_chunks("test", "a.md")

        # Interaction exists but chunk is gone
        conn = store._connect()
        count_before = conn.execute("SELECT COUNT(*) FROM chunk_interactions").fetchone()[0]
        assert count_before == 1

        cleaned = store.clean_orphaned_interactions()
        assert cleaned == 1

        count_after = conn.execute("SELECT COUNT(*) FROM chunk_interactions").fetchone()[0]
        assert count_after == 0

    def test_batch_log_interactions_with_file_info(self, db_path):
        """batch_log_interactions should accept and store project_id/file_path."""
        store, ids = _setup_project_with_chunks(db_path)
        store.batch_log_interactions(
            [
                (ids[0], "read", "Read", "test", "file0.md"),
                (ids[1], "edit", "Edit", "test", "file1.md"),
            ]
        )
        conn = store._connect()
        rows = conn.execute(
            "SELECT chunk_id, project_id, file_path, interaction "
            "FROM chunk_interactions ORDER BY chunk_id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["project_id"] == "test"
        assert rows[0]["file_path"] == "file0.md"
        assert rows[1]["interaction"] == "edit"


def _seed_interactions(store, project_id="test"):
    """Seed interaction and search data for trace tests."""
    store.register_project(project_id, f"/tmp/{project_id}", "content")
    ids = store.insert_chunks(
        project_id,
        [
            {"file_path": "auth.py", "content": "auth code"},
            {"file_path": "db.py", "content": "db code"},
        ],
    )
    # auth.py: 5 reads, 2 edits, 3 search_hits = 10 total
    # db.py: 2 reads = 2 total
    store.batch_log_interactions(
        [(ids[0], "read", "Read", project_id, "auth.py")] * 5
        + [(ids[0], "edit", "Edit", project_id, "auth.py")] * 2
        + [(ids[0], "search_hit", "search", project_id, "auth.py")] * 3
        + [(ids[1], "read", "Read", project_id, "db.py")] * 2
    )

    # Search logs: "jwt auth" found 3 times, "websocket" not found 2 times
    store.log_search("jwt auth", project_id, "all", [ids[0]], 1)
    store.log_search("jwt auth", project_id, "all", [ids[0]], 1)
    store.log_search("jwt auth", project_id, "all", [ids[0]], 1)
    store.log_search("websocket", project_id, "all", [], 0)
    store.log_search("websocket", project_id, "all", [], 0)
    store.log_search("rate limit", project_id, "all", [], 0)

    return ids


class TestTraceAggregations:
    def test_get_hot_chunks(self, db_path):
        store = Store(db_path)
        _seed_interactions(store)
        rows = store.get_hot_chunks(project_id="test", days=30, limit=10)
        assert len(rows) == 2
        assert rows[0]["file_path"] == "auth.py"
        assert rows[0]["total"] == 10
        assert rows[0]["reads"] == 5
        assert rows[0]["edits"] == 2
        assert rows[0]["searches"] == 3
        assert rows[1]["file_path"] == "db.py"
        assert rows[1]["total"] == 2

    def test_get_hot_chunks_no_project_filter(self, db_path):
        store = Store(db_path)
        _seed_interactions(store)
        rows = store.get_hot_chunks(project_id=None, days=30, limit=10)
        assert len(rows) == 2

    def test_get_knowledge_gaps(self, db_path):
        store = Store(db_path)
        _seed_interactions(store)
        rows = store.get_knowledge_gaps(project_id="test", days=30, limit=10)
        assert len(rows) == 2
        assert rows[0]["query"] == "websocket"
        assert rows[0]["count"] == 2
        assert rows[1]["query"] == "rate limit"
        assert rows[1]["count"] == 1

    def test_get_knowledge_gaps_excludes_found(self, db_path):
        """Gaps should not include queries that returned results."""
        store = Store(db_path)
        _seed_interactions(store)
        rows = store.get_knowledge_gaps(project_id="test", days=30, limit=10)
        queries = [r["query"] for r in rows]
        assert "jwt auth" not in queries

    def test_get_decaying_chunks(self, db_path):
        """Decay requires interactions in previous window but not current."""
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        ids = store.insert_chunks("test", [{"file_path": "old.py", "content": "old"}])
        # 10 interactions 45 days ago (in prev window for days=30)
        conn = store._connect()
        for _ in range(10):
            store.log_interaction(ids[0], "read", "Read", "test", "old.py")
        conn.execute("UPDATE chunk_interactions SET timestamp = datetime('now', '-45 days')")
        conn.commit()

        rows = store.get_decaying_chunks(project_id="test", days=30, limit=10)
        assert len(rows) == 1
        assert rows[0]["file_path"] == "old.py"
        assert rows[0]["before_count"] == 10
        assert rows[0]["now_count"] == 0
        assert rows[0]["ratio"] == 10.0

    def test_get_decaying_chunks_active_excluded(self, db_path):
        """Files still active should not appear in decay."""
        store = Store(db_path)
        _seed_interactions(store)  # All interactions are recent
        rows = store.get_decaying_chunks(project_id="test", days=30, limit=10)
        assert len(rows) == 0

    def test_get_top_queries(self, db_path):
        store = Store(db_path)
        _seed_interactions(store)
        rows = store.get_top_queries(project_id="test", days=30, limit=10)
        assert len(rows) == 3
        assert rows[0]["query"] == "jwt auth"
        assert rows[0]["count"] == 3
        assert rows[0]["avg_results"] == 1.0
        assert rows[1]["query"] == "websocket"
        assert rows[1]["count"] == 2
        assert rows[1]["avg_results"] == 0.0

    def test_get_hot_chunks_empty(self, db_path):
        store = Store(db_path)
        rows = store.get_hot_chunks(days=30, limit=10)
        assert rows == []

    def test_get_knowledge_gaps_empty(self, db_path):
        store = Store(db_path)
        rows = store.get_knowledge_gaps(days=30, limit=10)
        assert rows == []
