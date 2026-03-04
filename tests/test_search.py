"""Tests for hybrid search."""

from __future__ import annotations

import pytest

from dnomia_knowledge.embedder import Embedder
from dnomia_knowledge.models import SearchResult
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


class TestInteractionBoost:
    def _make_result(self, chunk_id: int, score: float) -> SearchResult:
        return SearchResult(
            chunk_id=chunk_id,
            project_id="test",
            file_path=f"file{chunk_id}.md",
            chunk_domain="content",
            chunk_type="heading",
            score=score,
        )

    def test_max_boost_with_10_reads(self, populated_store):
        store, embedder = populated_store
        search = HybridSearch(store, embedder)

        chunk_ids = store.insert_chunks(
            "test",
            [
                {
                    "file_path": "boost.md",
                    "chunk_domain": "content",
                    "chunk_type": "heading",
                    "name": "Boosted",
                    "language": "md",
                    "content": "Boosted chunk content.",
                },
            ],
        )
        cid = chunk_ids[0]
        for _ in range(10):
            store.log_interaction(cid, "read", "test")

        results = [self._make_result(cid, 0.5)]
        boosted = search._apply_interaction_boost(results, "test")
        assert boosted[0].score == pytest.approx(0.6, abs=1e-6)

    def test_half_boost_with_5_interactions(self, populated_store):
        store, embedder = populated_store
        search = HybridSearch(store, embedder)

        chunk_ids = store.insert_chunks(
            "test",
            [
                {
                    "file_path": "half.md",
                    "chunk_domain": "content",
                    "chunk_type": "heading",
                    "name": "Half",
                    "language": "md",
                    "content": "Half boost chunk.",
                },
            ],
        )
        cid = chunk_ids[0]
        for _ in range(5):
            store.log_interaction(cid, "edit", "test")

        results = [self._make_result(cid, 0.5)]
        boosted = search._apply_interaction_boost(results, "test")
        assert boosted[0].score == pytest.approx(0.55, abs=1e-6)

    def test_no_boost_with_zero_interactions(self, populated_store):
        store, embedder = populated_store
        search = HybridSearch(store, embedder)

        chunk_ids = store.insert_chunks(
            "test",
            [
                {
                    "file_path": "zero.md",
                    "chunk_domain": "content",
                    "chunk_type": "heading",
                    "name": "Zero",
                    "language": "md",
                    "content": "No interactions chunk.",
                },
            ],
        )
        cid = chunk_ids[0]

        results = [self._make_result(cid, 0.5)]
        boosted = search._apply_interaction_boost(results, "test")
        assert boosted[0].score == pytest.approx(0.5, abs=1e-6)

    def test_search_hit_does_not_affect_boost(self, populated_store):
        store, embedder = populated_store
        search = HybridSearch(store, embedder)

        chunk_ids = store.insert_chunks(
            "test",
            [
                {
                    "file_path": "nohit.md",
                    "chunk_domain": "content",
                    "chunk_type": "heading",
                    "name": "NoHit",
                    "language": "md",
                    "content": "Search hit only chunk.",
                },
            ],
        )
        cid = chunk_ids[0]
        for _ in range(10):
            store.log_interaction(cid, "search_hit", "search")

        results = [self._make_result(cid, 0.5)]
        boosted = search._apply_interaction_boost(results, "test")
        assert boosted[0].score == pytest.approx(0.5, abs=1e-6)

    def test_boost_resorts_results(self, populated_store):
        store, embedder = populated_store
        search = HybridSearch(store, embedder)

        chunks_data = [
            {
                "file_path": "high.md",
                "chunk_domain": "content",
                "chunk_type": "heading",
                "name": "High",
                "language": "md",
                "content": "Originally higher scored.",
            },
            {
                "file_path": "low.md",
                "chunk_domain": "content",
                "chunk_type": "heading",
                "name": "Low",
                "language": "md",
                "content": "Originally lower scored but many reads.",
            },
        ]
        chunk_ids = store.insert_chunks("test", chunks_data)
        high_id, low_id = chunk_ids

        for _ in range(10):
            store.log_interaction(low_id, "read", "test")

        results = [
            self._make_result(high_id, 0.5),
            self._make_result(low_id, 0.45),
        ]
        boosted = search._apply_interaction_boost(results, "test")
        assert boosted[0].chunk_id == low_id
        assert boosted[1].chunk_id == high_id


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
