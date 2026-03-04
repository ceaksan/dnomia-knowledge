"""Tests for hybrid search."""

from __future__ import annotations

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


class TestSearchLogging:
    def test_search_creates_search_log_entry(self, populated_store):
        store, embedder = populated_store
        search = HybridSearch(store, embedder)
        search.search("JWT authentication", project_id="test")

        logs = store.get_search_log(project_id="test")
        assert len(logs) >= 1
        assert logs[0]["query"] == "JWT authentication"
        assert logs[0]["project_id"] == "test"

    def test_search_logs_search_hit_interactions(self, populated_store):
        store, embedder = populated_store
        search = HybridSearch(store, embedder)
        results = search.search("JWT authentication", project_id="test")
        assert len(results) > 0

        chunk_ids = [r.chunk_id for r in results]
        counts = store.get_interaction_counts(chunk_ids, interactions=["search_hit"])
        assert sum(counts.values()) > 0

    def test_empty_search_no_log(self, populated_store):
        store, embedder = populated_store
        search = HybridSearch(store, embedder)
        search.search("", project_id="test")

        logs = store.get_search_log(project_id="test")
        assert len(logs) == 0

    def test_search_log_stores_result_count(self, populated_store):
        store, embedder = populated_store
        search = HybridSearch(store, embedder)
        results = search.search("authentication", project_id="test")

        logs = store.get_search_log(project_id="test")
        assert logs[0]["result_count"] == len(results)


class TestRRFMerge:
    def test_rrf_basic(self):
        from dnomia_knowledge.models import SearchResult
        from dnomia_knowledge.search import rrf_merge

        list_a = [
            SearchResult(
                chunk_id=1,
                project_id="t",
                file_path="a",
                chunk_domain="c",
                chunk_type="h",
                score=0.9,
            ),
            SearchResult(
                chunk_id=2,
                project_id="t",
                file_path="b",
                chunk_domain="c",
                chunk_type="h",
                score=0.8,
            ),
        ]
        list_b = [
            SearchResult(
                chunk_id=2,
                project_id="t",
                file_path="b",
                chunk_domain="c",
                chunk_type="h",
                score=0.7,
            ),
            SearchResult(
                chunk_id=3,
                project_id="t",
                file_path="c",
                chunk_domain="c",
                chunk_type="h",
                score=0.6,
            ),
        ]
        merged = rrf_merge(list_a, list_b, k=60)
        # chunk_id=2 appears in both lists, should rank highest
        assert merged[0].chunk_id == 2

    def test_rrf_respects_limit(self):
        from dnomia_knowledge.models import SearchResult
        from dnomia_knowledge.search import rrf_merge

        list_a = [
            SearchResult(
                chunk_id=i, project_id="t", file_path=f"{i}", chunk_domain="c", chunk_type="h"
            )
            for i in range(10)
        ]
        merged = rrf_merge(list_a, [], k=60, limit=3)
        assert len(merged) == 3
