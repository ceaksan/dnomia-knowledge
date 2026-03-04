"""Tests for GraphBuilder."""

from __future__ import annotations

import json
import math

from dnomia_knowledge.graph import GraphBuilder
from dnomia_knowledge.registry import GraphConfig
from dnomia_knowledge.store import Store


def _make_store(db_path: str) -> Store:
    store = Store(db_path)
    store.register_project("test", "/tmp/test", "content", graph_enabled=True)
    return store


def _enabled_config(**overrides) -> GraphConfig:
    defaults = {
        "enabled": True,
        "edge_types": ["link", "tag", "category", "semantic", "import"],
        "semantic_threshold": 0.75,
    }
    defaults.update(overrides)
    return GraphConfig(**defaults)


def _insert_chunks_with_meta(store, metas: list[dict], file_path: str = "doc.md") -> list[int]:
    """Insert chunks with given metadata dicts, return chunk IDs."""
    chunks = []
    for i, meta in enumerate(metas):
        chunks.append(
            {
                "file_path": file_path,
                "chunk_domain": "content",
                "chunk_type": "block",
                "content": f"Chunk {i} content",
                "metadata": json.dumps(meta),
            }
        )
    return store.insert_chunks("test", chunks)


def _make_normalized_vector(dim: int = 768, seed_val: float = 0.0) -> list[float]:
    """Create a normalized vector of given dimension."""
    vec = [seed_val + (i * 0.01) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


class TestTagEdges:
    def test_two_shared_tags_creates_edge(self, db_path):
        store = _make_store(db_path)
        config = _enabled_config()
        gb = GraphBuilder(store, config)

        ids = _insert_chunks_with_meta(
            store,
            [
                {"tags": ["python", "testing", "web"]},
                {"tags": ["python", "testing", "api"]},
            ],
        )

        count = gb.build_edges_for_file("test", "doc.md", ids)
        assert count >= 1

        edges = store.get_edges_for_project("test")
        tag_edges = [e for e in edges if e["edge_type"] == "tag"]
        assert len(tag_edges) == 1
        assert 0 < tag_edges[0]["weight"] <= 1.0

    def test_one_shared_tag_no_edge(self, db_path):
        store = _make_store(db_path)
        config = _enabled_config()
        gb = GraphBuilder(store, config)

        ids = _insert_chunks_with_meta(
            store,
            [
                {"tags": ["python", "web"]},
                {"tags": ["python", "api"]},
            ],
        )

        gb.build_edges_for_file("test", "doc.md", ids)

        edges = store.get_edges_for_project("test")
        tag_edges = [e for e in edges if e["edge_type"] == "tag"]
        assert len(tag_edges) == 0


class TestCategoryEdges:
    def test_same_category_creates_edge(self, db_path):
        store = _make_store(db_path)
        config = _enabled_config()
        gb = GraphBuilder(store, config)

        ids = _insert_chunks_with_meta(
            store,
            [
                {"categories": ["tutorial"]},
                {"categories": ["tutorial"]},
            ],
        )

        gb.build_edges_for_file("test", "doc.md", ids)

        edges = store.get_edges_for_project("test")
        cat_edges = [e for e in edges if e["edge_type"] == "category"]
        assert len(cat_edges) == 1


class TestSemanticEdges:
    def test_semantic_edges_with_similar_vectors(self, db_path):
        store = _make_store(db_path)
        config = _enabled_config(semantic_threshold=0.5)
        gb = GraphBuilder(store, config)

        ids = _insert_chunks_with_meta(store, [{"tags": []}, {"tags": []}, {"tags": []}])

        # Create very similar vectors
        v1 = _make_normalized_vector(768, 1.0)
        v2 = _make_normalized_vector(768, 1.001)
        v3 = _make_normalized_vector(768, 1.002)
        store.insert_chunk_vectors(ids, [v1, v2, v3])

        count = gb.build_semantic_edges("test")
        assert count > 0

    def test_threshold_filtering(self, db_path):
        store = _make_store(db_path)
        # Very high threshold - nothing should pass
        config = _enabled_config(semantic_threshold=0.9999)
        gb = GraphBuilder(store, config)

        ids = _insert_chunks_with_meta(store, [{"tags": []}, {"tags": []}])

        # Create somewhat different vectors
        v1 = _make_normalized_vector(768, 0.0)
        v2 = _make_normalized_vector(768, 5.0)
        store.insert_chunk_vectors(ids, [v1, v2])

        count = gb.build_semantic_edges("test")
        assert count == 0

    def test_semantic_edges_none_chunk_ids_processes_all(self, db_path):
        store = _make_store(db_path)
        config = _enabled_config(semantic_threshold=0.5)
        gb = GraphBuilder(store, config)

        ids = _insert_chunks_with_meta(store, [{"tags": []}, {"tags": []}])
        v1 = _make_normalized_vector(768, 1.0)
        v2 = _make_normalized_vector(768, 1.001)
        store.insert_chunk_vectors(ids, [v1, v2])

        # Pass None -> processes all chunks in project
        count = gb.build_semantic_edges("test", chunk_ids=None)
        assert count > 0


class TestLinkEdges:
    def test_markdown_link_creates_edge(self, db_path):
        store = _make_store(db_path)
        config = _enabled_config()
        gb = GraphBuilder(store, config)

        # Create target chunk in another file
        target_ids = store.insert_chunks(
            "test",
            [
                {
                    "file_path": "docs/guide.md",
                    "chunk_domain": "content",
                    "chunk_type": "block",
                    "content": "Guide content",
                }
            ],
        )

        # Create source chunk with markdown link
        source_ids = store.insert_chunks(
            "test",
            [
                {
                    "file_path": "docs/index.md",
                    "chunk_domain": "content",
                    "chunk_type": "block",
                    "content": "See the [guide](guide.md) for details.",
                }
            ],
        )

        count = gb.build_edges_for_file("test", "docs/index.md", source_ids)
        assert count >= 1

        edges = store.get_edges_for_project("test")
        link_edges = [e for e in edges if e["edge_type"] == "link"]
        assert len(link_edges) == 1
        assert link_edges[0]["source_id"] == source_ids[0]
        assert link_edges[0]["target_id"] == target_ids[0]

    def test_external_links_ignored(self, db_path):
        store = _make_store(db_path)
        config = _enabled_config()
        gb = GraphBuilder(store, config)

        ids = store.insert_chunks(
            "test",
            [
                {
                    "file_path": "doc.md",
                    "chunk_domain": "content",
                    "chunk_type": "block",
                    "content": "Visit [site](https://example.com) or [anchor](#section)",
                }
            ],
        )

        count = gb.build_edges_for_file("test", "doc.md", ids)
        link_edges = [e for e in store.get_edges_for_project("test") if e["edge_type"] == "link"]
        assert len(link_edges) == 0
        assert count == 0


class TestImportEdges:
    def test_python_import_creates_edge(self, db_path):
        store = _make_store(db_path)
        config = _enabled_config()
        gb = GraphBuilder(store, config)

        # Target file
        store.insert_chunks(
            "test",
            [
                {
                    "file_path": "src/utils.py",
                    "chunk_domain": "code",
                    "chunk_type": "function",
                    "content": "def helper(): pass",
                }
            ],
        )

        # Source file with import
        source_ids = store.insert_chunks(
            "test",
            [
                {
                    "file_path": "src/main.py",
                    "chunk_domain": "code",
                    "chunk_type": "function",
                    "content": "from utils import helper\ndef run(): helper()",
                }
            ],
        )

        count = gb.build_import_edges("test", "src/main.py", source_ids)
        assert count >= 1

    def test_js_import_creates_edge(self, db_path):
        store = _make_store(db_path)
        config = _enabled_config()
        gb = GraphBuilder(store, config)

        # Target file
        store.insert_chunks(
            "test",
            [
                {
                    "file_path": "src/utils.ts",
                    "chunk_domain": "code",
                    "chunk_type": "function",
                    "content": "export function helper() {}",
                }
            ],
        )

        # Source file with import
        source_ids = store.insert_chunks(
            "test",
            [
                {
                    "file_path": "src/main.ts",
                    "chunk_domain": "code",
                    "chunk_type": "function",
                    "content": "import { helper } from './utils'",
                }
            ],
        )

        count = gb.build_import_edges("test", "src/main.ts", source_ids)
        assert count >= 1


class TestCommunityDetection:
    def test_louvain_produces_valid_partition(self, db_path):
        store = _make_store(db_path)
        config = _enabled_config()
        gb = GraphBuilder(store, config)

        ids = _insert_chunks_with_meta(
            store, [{"tags": []}, {"tags": []}, {"tags": []}, {"tags": []}]
        )

        # Create edges forming two clusters
        store.insert_edges(
            [
                {"source_id": ids[0], "target_id": ids[1], "edge_type": "tag", "weight": 1.0},
                {"source_id": ids[2], "target_id": ids[3], "edge_type": "tag", "weight": 1.0},
            ]
        )

        n_communities = gb.run_community_detection("test")
        assert n_communities >= 1

        # Verify all chunks got a community_id
        conn = store._connect()
        for cid in ids:
            row = conn.execute("SELECT metadata FROM chunks WHERE id = ?", (cid,)).fetchone()
            meta = json.loads(row[0])
            assert "community_id" in meta
            assert isinstance(meta["community_id"], int)

    def test_pagerank_values_positive(self, db_path):
        store = _make_store(db_path)
        config = _enabled_config()
        gb = GraphBuilder(store, config)

        ids = _insert_chunks_with_meta(store, [{"tags": []}, {"tags": []}, {"tags": []}])

        store.insert_edges(
            [
                {"source_id": ids[0], "target_id": ids[1], "edge_type": "link", "weight": 1.0},
                {"source_id": ids[1], "target_id": ids[2], "edge_type": "link", "weight": 1.0},
            ]
        )

        gb.run_community_detection("test")

        conn = store._connect()
        for cid in ids:
            row = conn.execute("SELECT metadata FROM chunks WHERE id = ?", (cid,)).fetchone()
            meta = json.loads(row[0])
            assert "pagerank" in meta
            assert isinstance(meta["pagerank"], float)
            assert meta["pagerank"] > 0


class TestRebuildAllEdges:
    def test_rebuild_clears_old_and_recreates(self, db_path):
        store = _make_store(db_path)
        config = _enabled_config()
        gb = GraphBuilder(store, config)

        ids = _insert_chunks_with_meta(
            store,
            [
                {"tags": ["a", "b", "c"], "categories": ["tutorial"]},
                {"tags": ["a", "b", "d"], "categories": ["tutorial"]},
            ],
        )

        # Insert a stale edge
        store.insert_edges(
            [{"source_id": ids[0], "target_id": ids[1], "edge_type": "stale", "weight": 1.0}]
        )

        counts = gb.rebuild_all_edges("test")
        assert isinstance(counts, dict)
        assert counts["tag"] >= 1
        assert counts["category"] >= 1

        # Stale edges should be gone
        edges = store.get_edges_for_project("test")
        stale = [e for e in edges if e["edge_type"] == "stale"]
        assert len(stale) == 0


class TestDisabledGraph:
    def test_disabled_returns_zero(self, db_path):
        store = _make_store(db_path)
        config = GraphConfig(enabled=False)
        gb = GraphBuilder(store, config)

        ids = _insert_chunks_with_meta(store, [{"tags": ["a", "b"]}, {"tags": ["a", "b"]}])

        assert gb.build_edges_for_file("test", "doc.md", ids) == 0
        assert gb.build_semantic_edges("test") == 0
        assert gb.build_import_edges("test", "doc.md", ids) == 0
        assert gb.rebuild_all_edges("test") == {}
        assert gb.run_community_detection("test") == 0


class TestEmptyProject:
    def test_empty_project_no_crash(self, db_path):
        store = _make_store(db_path)
        config = _enabled_config()
        gb = GraphBuilder(store, config)

        assert gb.build_edges_for_file("test", "doc.md", []) == 0
        assert gb.build_semantic_edges("test") == 0
        assert gb.build_import_edges("test", "doc.md", []) == 0
        counts = gb.rebuild_all_edges("test")
        assert all(v == 0 for v in counts.values())
        assert gb.run_community_detection("test") == 0
