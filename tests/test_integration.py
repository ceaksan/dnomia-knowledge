"""Integration tests for Sprint 2, Sprint 3, Sprint 4, and Sprint 5."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from dnomia_knowledge.embedder import Embedder
from dnomia_knowledge.graph import GraphBuilder
from dnomia_knowledge.indexer import Indexer
from dnomia_knowledge.registry import load_config
from dnomia_knowledge.search import HybridSearch
from dnomia_knowledge.store import Store


@pytest.fixture
def full_project(tmp_dir):
    """Create a project with .knowledge.toml, content, and code files."""
    toml_content = b"""
[project]
name = "test-project"
type = "saas"

[content]
paths = ["docs/"]
extensions = [".md"]

[code]
preset = "python"
paths = ["src/"]

[indexing]
ignore_patterns = ["__pycache__"]
"""
    (tmp_dir / ".knowledge.toml").write_bytes(toml_content)
    (tmp_dir / "docs").mkdir()
    (tmp_dir / "docs" / "guide.md").write_text(
        "## Authentication Guide\n\n"
        "This guide covers JWT token authentication for the application. "
        "JSON Web Tokens provide a secure way to transmit information between "
        "parties as a JSON object. Use refresh tokens for session management. "
        "Always validate tokens on the server side before granting access."
    )
    (tmp_dir / "src").mkdir()
    (tmp_dir / "src" / "auth.py").write_text(
        "def validate_token(token: str) -> bool:\n"
        '    """Validate JWT token."""\n'
        "    if not token:\n"
        "        return False\n"
        "    parts = token.split('.')\n"
        "    return len(parts) == 3\n"
    )
    return tmp_dir


class TestFullPipeline:
    def test_index_with_config(self, full_project, db_path):
        """Index a project with .knowledge.toml - both content and code files."""
        config = load_config(full_project)
        assert config is not None
        assert config.name == "test-project"

        store = Store(db_path)
        embedder = Embedder()
        indexer = Indexer(store, embedder)

        result = indexer.index_directory(
            project_id=config.name,
            directory=str(full_project),
            config=config,
        )

        assert result.total_files == 2
        assert result.content_chunks > 0
        assert result.code_chunks > 0
        assert result.status == "completed"
        store.close()

    def test_search_content_and_code(self, full_project, db_path):
        """Search should find results from both content and code chunks."""
        config = load_config(full_project)
        store = Store(db_path)
        embedder = Embedder()
        indexer = Indexer(store, embedder)
        indexer.index_directory("test-project", str(full_project), config=config)

        search = HybridSearch(store, embedder)

        results = search.search("JWT authentication", project_id="test-project")
        assert len(results) > 0

        store.close()

    def test_backward_compat_no_config(self, tmp_dir, db_path, sample_markdown):
        """Without .knowledge.toml, works exactly like Sprint 1."""
        (tmp_dir / "post.md").write_text(sample_markdown)
        (tmp_dir / "app.py").write_text("def hello(): pass\n")

        store = Store(db_path)
        embedder = Embedder()
        indexer = Indexer(store, embedder)

        result = indexer.index_directory("test", str(tmp_dir))
        assert result.total_files == 1  # only .md, .py ignored
        assert result.code_chunks == 0
        store.close()

    def test_domain_filter_code(self, full_project, db_path):
        """Search with domain='code' should return only code chunks."""
        config = load_config(full_project)
        store = Store(db_path)
        embedder = Embedder()
        indexer = Indexer(store, embedder)
        indexer.index_directory("test-project", str(full_project), config=config)

        search = HybridSearch(store, embedder)

        code_results = search.search("validate token", project_id="test-project", domain="code")
        for r in code_results:
            assert r.chunk_domain == "code"

        store.close()


class TestSprint3Integration:
    """Sprint 3: AST chunking, knowledge graph, cross-project search."""

    @pytest.fixture
    def python_project(self, tmp_dir):
        """Project with Python code for AST chunking tests."""
        toml = b"""
[project]
name = "ast-test"
type = "saas"

[code]
preset = "python"
paths = ["src/"]
"""
        (tmp_dir / ".knowledge.toml").write_bytes(toml)
        (tmp_dir / "src").mkdir()
        (tmp_dir / "src" / "utils.py").write_text(
            "def compute_total(items: list) -> float:\n"
            '    """Compute total price of items."""\n'
            "    return sum(i.price for i in items)\n"
            "\n\n"
            "def format_currency(amount: float) -> str:\n"
            '    """Format amount as currency string."""\n'
            '    return f"${amount:.2f}"\n'
        )
        return tmp_dir

    @pytest.fixture
    def graph_project(self, tmp_dir):
        """Project with graph enabled and tagged content for edge creation.

        Each file has multiple heading sections (>200 chars each) so the
        MdChunker produces multiple chunks per file, all sharing the same
        frontmatter tags. This lets _build_tag_edges create edges.
        """
        toml = b"""
[project]
name = "graph-test"
type = "saas"

[content]
paths = ["docs/"]
extensions = [".md"]

[graph]
enabled = true
edge_types = ["tag", "category"]
"""
        (tmp_dir / ".knowledge.toml").write_bytes(toml)
        (tmp_dir / "docs").mkdir()
        (tmp_dir / "docs" / "auth.md").write_text(
            "---\n"
            "title: Authentication\n"
            "tags: [security, jwt, auth]\n"
            "categories: [backend]\n"
            "---\n\n"
            "## Token Validation\n\n"
            "JWT authentication handles user sessions securely by validating "
            "tokens on every request. The server decodes the token payload, "
            "checks the signature against the secret key, and verifies that "
            "the token has not expired. If any of these checks fail, the "
            "request is rejected with a 401 Unauthorized response. This "
            "ensures that only legitimate users can access protected "
            "resources and endpoints in the application.\n\n"
            "## Session Management\n\n"
            "Refresh tokens allow users to maintain their sessions without "
            "re-entering credentials. When the access token expires, the "
            "client sends the refresh token to obtain a new access token. "
            "The server validates the refresh token, checks it against the "
            "stored token family, and issues a new token pair. Token rotation "
            "prevents replay attacks by invalidating old refresh tokens "
            "whenever a new pair is issued to the client.\n"
        )
        return tmp_dir

    def test_ast_chunks_in_pipeline(self, python_project, db_path):
        """AST chunker produces function-level chunks, not raw blocks."""
        config = load_config(python_project)
        store = Store(db_path)
        embedder = Embedder()
        indexer = Indexer(store, embedder)

        result = indexer.index_directory(
            project_id=config.name,
            directory=str(python_project),
            config=config,
        )

        assert result.code_chunks >= 2

        conn = store._connect()
        rows = conn.execute(
            "SELECT chunk_type FROM chunks WHERE project_id = ?",
            (config.name,),
        ).fetchall()

        chunk_types = {r[0] for r in rows}
        assert "function" in chunk_types
        assert "block" not in chunk_types

        store.close()

    def test_graph_edges_created_on_index(self, graph_project, db_path):
        """Indexing with graph.enabled=true creates edges in the edges table."""
        config = load_config(graph_project)
        store = Store(db_path)
        embedder = Embedder()
        indexer = Indexer(store, embedder)

        indexer.index_directory(
            project_id=config.name,
            directory=str(graph_project),
            config=config,
        )

        edges = store.get_edges_for_project(config.name)
        assert len(edges) > 0

        edge_types = {e["edge_type"] for e in edges}
        assert "tag" in edge_types

        store.close()

    def test_rebuild_graph_creates_communities(self, graph_project, db_path):
        """rebuild_all_edges + run_community_detection writes community_id to metadata."""
        config = load_config(graph_project)
        store = Store(db_path)
        embedder = Embedder()
        indexer = Indexer(store, embedder)

        indexer.index_directory(
            project_id=config.name,
            directory=str(graph_project),
            config=config,
        )

        builder = GraphBuilder(store, config.graph)
        builder.rebuild_all_edges(config.name)
        num_communities = builder.run_community_detection(config.name)

        assert num_communities >= 1

        conn = store._connect()
        rows = conn.execute(
            "SELECT metadata FROM chunks WHERE project_id = ?",
            (config.name,),
        ).fetchall()

        for row in rows:
            meta = json.loads(row[0]) if row[0] else {}
            assert "community_id" in meta
            assert "pagerank" in meta

        store.close()

    def test_graph_query_neighbors(self, db_path):
        """get_neighbors traverses edges and returns correct neighbors."""
        store = Store(db_path)
        store.register_project("neighbor-test", "/tmp/n", "content")

        ids = store.insert_chunks(
            "neighbor-test",
            [
                {"file_path": "a.md", "content": "chunk A"},
                {"file_path": "b.md", "content": "chunk B"},
                {"file_path": "c.md", "content": "chunk C"},
            ],
        )

        store.insert_edges(
            [
                {"source_id": ids[0], "target_id": ids[1], "edge_type": "tag", "weight": 0.8},
                {"source_id": ids[1], "target_id": ids[2], "edge_type": "tag", "weight": 0.6},
            ]
        )

        neighbors = store.get_neighbors(ids[0], depth=1)
        neighbor_ids = {n["chunk_id"] for n in neighbors}
        assert ids[1] in neighbor_ids
        assert ids[2] not in neighbor_ids

        neighbors_d2 = store.get_neighbors(ids[0], depth=2)
        neighbor_ids_d2 = {n["chunk_id"] for n in neighbors_d2}
        assert ids[1] in neighbor_ids_d2
        assert ids[2] in neighbor_ids_d2

        store.close()

    def test_cross_project_search(self, db_path):
        """search_cross returns results from both primary and related projects."""
        store = Store(db_path)
        embedder = Embedder()
        indexer = Indexer(store, embedder)

        with tempfile.TemporaryDirectory() as dir_a, tempfile.TemporaryDirectory() as dir_b:
            path_a = Path(dir_a)
            path_b = Path(dir_b)

            toml_a = b"""
[project]
name = "project-a"
type = "content"

[content]
paths = ["docs/"]
extensions = [".md"]
"""
            (path_a / ".knowledge.toml").write_bytes(toml_a)
            (path_a / "docs").mkdir()
            (path_a / "docs" / "deploy.md").write_text(
                "## Deployment Guide\n\n"
                "This guide covers deployment to production servers "
                "using Docker containers and orchestration tools. "
                "Continuous deployment pipelines automate the release "
                "process for faster delivery cycles.\n"
            )

            toml_b = b"""
[project]
name = "project-b"
type = "content"

[content]
paths = ["docs/"]
extensions = [".md"]
"""
            (path_b / ".knowledge.toml").write_bytes(toml_b)
            (path_b / "docs").mkdir()
            (path_b / "docs" / "infra.md").write_text(
                "## Infrastructure\n\n"
                "Production infrastructure runs on Kubernetes clusters. "
                "Docker containers are deployed via Helm charts. "
                "Monitoring and alerting ensure system reliability "
                "across all deployment environments.\n"
            )

            config_a = load_config(path_a)
            config_b = load_config(path_b)

            indexer.index_directory("project-a", str(path_a), config=config_a)
            indexer.index_directory("project-b", str(path_b), config=config_b)

        search = HybridSearch(store, embedder)
        results = search.search_cross(
            query="deployment Docker",
            project_id="project-a",
            related_projects=["project-b"],
        )

        assert len(results) > 0
        project_ids = {r.project_id for r in results}
        assert len(project_ids) >= 2

        store.close()

    def test_cross_project_no_related_is_regular_search(self, db_path):
        """search_cross with empty related list behaves like regular search."""
        store = Store(db_path)
        embedder = Embedder()
        indexer = Indexer(store, embedder)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            toml = b"""
[project]
name = "solo-project"
type = "content"

[content]
paths = ["docs/"]
extensions = [".md"]
"""
            (path / ".knowledge.toml").write_bytes(toml)
            (path / "docs").mkdir()
            (path / "docs" / "api.md").write_text(
                "## API Reference\n\n"
                "The REST API provides endpoints for managing resources. "
                "All endpoints require authentication via bearer tokens. "
                "Rate limiting is applied per API key to prevent abuse.\n"
            )

            config = load_config(path)
            indexer.index_directory("solo-project", str(path), config=config)

        search = HybridSearch(store, embedder)

        regular = search.search("API authentication", project_id="solo-project")
        cross = search.search_cross(
            query="API authentication",
            project_id="solo-project",
            related_projects=[],
        )

        assert len(regular) > 0
        assert len(cross) > 0

        regular_ids = {r.chunk_id for r in regular}
        cross_ids = {r.chunk_id for r in cross}
        assert regular_ids == cross_ids

        store.close()


class TestSprint4Integration:
    """Sprint 4: interactions, search logging, hook script, GC."""

    @pytest.fixture
    def indexed_project(self, tmp_dir, db_path):
        """Index a project with content + code files, return (store, project_id, tmp_dir)."""
        toml_content = b"""
[project]
name = "s4-test"
type = "saas"

[content]
paths = ["docs/"]
extensions = [".md"]

[code]
preset = "python"
paths = ["src/"]
"""
        (tmp_dir / ".knowledge.toml").write_bytes(toml_content)
        (tmp_dir / "docs").mkdir()
        (tmp_dir / "docs" / "guide.md").write_text(
            "## Caching Strategy\n\n"
            "Redis caching improves application performance significantly. "
            "Cache invalidation must be handled carefully to avoid stale data. "
            "Use TTL-based expiration for session data and event-driven "
            "invalidation for mutable resources. Always monitor cache hit rates "
            "to ensure the caching layer is effective.\n"
        )
        (tmp_dir / "src").mkdir()
        (tmp_dir / "src" / "cache.py").write_text(
            "def get_cached(key: str) -> str | None:\n"
            '    """Get value from cache by key."""\n'
            "    return None\n"
            "\n\n"
            "def set_cached(key: str, value: str, ttl: int = 300) -> None:\n"
            '    """Set value in cache with TTL."""\n'
            "    pass\n"
        )

        config = load_config(tmp_dir)
        store = Store(db_path)
        embedder = Embedder()
        indexer = Indexer(store, embedder)
        indexer.index_directory("s4-test", str(tmp_dir), config=config)

        yield store, "s4-test", tmp_dir, embedder
        store.close()

    def test_index_search_logs_entry(self, indexed_project):
        """Index project, search, verify search_log entry exists."""
        store, project_id, _, embedder = indexed_project
        search = HybridSearch(store, embedder)

        results = search.search("caching strategy", project_id=project_id)
        assert len(results) > 0

        logs = store.get_search_log(project_id=project_id)
        assert len(logs) >= 1

        matching = [log for log in logs if log["query"] == "caching strategy"]
        assert len(matching) == 1
        assert matching[0]["project_id"] == project_id
        assert matching[0]["result_count"] > 0

        result_ids = json.loads(matching[0]["result_chunk_ids"])
        assert len(result_ids) > 0

    def test_interaction_boost_ranks_higher(self, db_path):
        """Log interactions for one chunk, verify it scores higher after boost."""
        store = Store(db_path)
        embedder = Embedder()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            toml = b"""
[project]
name = "boost-test"
type = "content"

[content]
paths = ["docs/"]
extensions = [".md"]
"""
            (path / ".knowledge.toml").write_bytes(toml)
            (path / "docs").mkdir()
            # Two docs about similar topic so both appear in results
            (path / "docs" / "redis.md").write_text(
                "## Redis Caching\n\n"
                "Redis provides in-memory data structure store used as database "
                "and cache. Redis supports various data structures including "
                "strings, hashes, lists, sets, and sorted sets. Performance of "
                "Redis caching depends on proper key design and eviction policy.\n"
            )
            (path / "docs" / "memcached.md").write_text(
                "## Memcached Caching\n\n"
                "Memcached is a distributed memory caching system designed for "
                "speed and simplicity. It caches data and objects in RAM to "
                "reduce database load. Memcached caching is widely used for "
                "session storage and frequently accessed data retrieval.\n"
            )

            config = load_config(path)
            indexer = Indexer(store, embedder)
            indexer.index_directory("boost-test", str(path), config=config)

        search = HybridSearch(store, embedder)

        # Search before any interactions to get baseline
        baseline = search.search("caching system performance", project_id="boost-test")
        assert len(baseline) >= 2

        # Find the chunk_id for redis.md
        redis_chunk = None
        memcached_chunk = None
        for r in baseline:
            if "redis" in r.file_path.lower():
                redis_chunk = r
            elif "memcached" in r.file_path.lower():
                memcached_chunk = r

        assert redis_chunk is not None
        assert memcached_chunk is not None

        # Log 10 read interactions for redis chunk
        for _ in range(10):
            store.log_interaction(redis_chunk.chunk_id, "read", "test")

        # Search again - redis chunk should have boost applied
        boosted = search.search("caching system performance", project_id="boost-test")
        assert len(boosted) >= 2

        # Find redis result in boosted search
        boosted_redis = None
        for r in boosted:
            if r.chunk_id == redis_chunk.chunk_id:
                boosted_redis = r
                break

        assert boosted_redis is not None

        # Verify the interaction counts exist for this chunk
        counts = store.get_interaction_counts(
            [redis_chunk.chunk_id], days=30, interactions=["read"]
        )
        assert counts.get(redis_chunk.chunk_id, 0) >= 10

        # Verify redis chunk has a higher score than memcached in boosted results
        boosted_memcached = None
        for r in boosted:
            if r.chunk_id == memcached_chunk.chunk_id:
                boosted_memcached = r
                break

        if boosted_memcached is not None:
            assert boosted_redis.score >= boosted_memcached.score

        store.close()

    def test_post_tool_use_hook_logs_interactions(self, db_path):
        """Run hook script as subprocess, verify interactions logged in DB."""
        store = Store(db_path)

        with tempfile.TemporaryDirectory() as td:
            project_path = td
            store.register_project("hook-test", project_path, "content")

            # Insert a chunk for a file within the project
            file_rel = "docs/api.md"
            ids = store.insert_chunks(
                "hook-test",
                [{"file_path": file_rel, "content": "API endpoint documentation."}],
            )
            assert len(ids) == 1

            # Build absolute file path that the hook will receive
            abs_file = os.path.join(project_path, file_rel)

            # Close store before subprocess uses the DB
            store.close()

            hook_input = json.dumps(
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": abs_file},
                }
            )

            result = subprocess.run(
                [sys.executable, "-m", "dnomia_knowledge.hooks.post_tool_use"],
                input=hook_input,
                capture_output=True,
                text=True,
                env={**os.environ, "DNOMIA_KNOWLEDGE_DB": db_path},
            )

            assert result.returncode == 0

            # Reopen store and verify interactions
            store2 = Store(db_path)
            counts = store2.get_interaction_counts(ids, days=1, interactions=["read"])
            assert counts.get(ids[0], 0) == 1
            store2.close()

    def test_gc_deletes_old_interactions_and_search_logs(self, db_path):
        """Insert old interactions/search_logs, GC removes them."""
        store = Store(db_path)
        store.register_project("gc-test", "/tmp/gc", "content")

        ids = store.insert_chunks(
            "gc-test",
            [{"file_path": "a.md", "content": "GC test chunk."}],
        )
        chunk_id = ids[0]

        # Log interactions and search entries
        store.log_interaction(chunk_id, "read", "test")
        store.log_interaction(chunk_id, "edit", "test")
        store.log_search("test query", "gc-test", "all", [chunk_id], 1)

        # Verify they exist
        counts_before = store.get_interaction_counts([chunk_id], days=1)
        assert counts_before.get(chunk_id, 0) == 2
        logs_before = store.get_search_log(project_id="gc-test")
        assert len(logs_before) == 1

        # Manually backdate timestamps to 100 days ago via raw SQL
        conn = store._connect()
        conn.execute(
            "UPDATE chunk_interactions SET timestamp = datetime('now', '-100 days') "
            "WHERE chunk_id = ?",
            (chunk_id,),
        )
        conn.execute(
            "UPDATE search_log SET timestamp = datetime('now', '-100 days') "
            "WHERE project_id = 'gc-test'",
        )
        conn.commit()

        # GC with 90-day threshold should remove them
        deleted_interactions = store.delete_old_interactions(90)
        assert deleted_interactions == 2

        deleted_logs = store.delete_old_search_logs(90)
        assert deleted_logs == 1

        # Verify they are gone
        counts_after = store.get_interaction_counts([chunk_id], days=365)
        assert counts_after.get(chunk_id, 0) == 0
        logs_after = store.get_search_log(project_id="gc-test")
        assert len(logs_after) == 0

        store.close()


class TestSprint5Integration:
    """Sprint 5: language/file_pattern filters, PreToolUse hook blocking."""

    @pytest.fixture
    def mixed_project(self, tmp_dir):
        """Project with .md and .py files in different directories."""
        toml_content = b"""
[project]
name = "s5-test"
type = "saas"

[content]
paths = ["docs/"]
extensions = [".md"]

[code]
preset = "python"
paths = ["src/"]
"""
        (tmp_dir / ".knowledge.toml").write_bytes(toml_content)
        (tmp_dir / "docs").mkdir()
        (tmp_dir / "docs" / "setup.md").write_text(
            "## Setup Guide\n\n"
            "This document explains how to set up the development environment "
            "for the project. Install Python dependencies with pip and configure "
            "environment variables before running the application. Make sure to "
            "use a virtual environment to isolate project dependencies from "
            "system packages.\n"
        )
        (tmp_dir / "docs" / "api.md").write_text(
            "## API Reference\n\n"
            "The REST API provides endpoints for managing user resources. "
            "Authentication is handled via bearer tokens. All requests must "
            "include an Authorization header with a valid JWT token. Rate "
            "limiting is applied per API key to prevent abuse and ensure "
            "fair usage across all consumers.\n"
        )
        (tmp_dir / "src").mkdir()
        (tmp_dir / "src" / "main.py").write_text(
            "def start_server(host: str, port: int) -> None:\n"
            '    """Start the HTTP server."""\n'
            "    print(f'Listening on {host}:{port}')\n"
            "\n\n"
            "def stop_server() -> None:\n"
            '    """Stop the HTTP server gracefully."""\n'
            "    print('Shutting down')\n"
        )
        (tmp_dir / "src" / "db.py").write_text(
            "def connect(dsn: str) -> object:\n"
            '    """Connect to the database."""\n'
            "    return None\n"
            "\n\n"
            "def disconnect(conn: object) -> None:\n"
            '    """Close database connection."""\n'
            "    pass\n"
        )
        return tmp_dir

    def test_search_with_language_filter(self, mixed_project, db_path):
        """Index project, search with language='python', verify only Python results."""
        config = load_config(mixed_project)
        store = Store(db_path)
        embedder = Embedder()
        indexer = Indexer(store, embedder)
        indexer.index_directory("s5-test", str(mixed_project), config=config)

        search = HybridSearch(store, embedder)
        results = search.search("server database", project_id="s5-test", language="python")

        assert len(results) > 0
        for r in results:
            assert r.language == "python", (
                f"Expected language='python', got '{r.language}' for {r.file_path}"
            )

        store.close()

    def test_search_with_file_pattern_filter(self, mixed_project, db_path):
        """Index project, search with file_pattern='docs', verify path filtering."""
        config = load_config(mixed_project)
        store = Store(db_path)
        embedder = Embedder()
        indexer = Indexer(store, embedder)
        indexer.index_directory("s5-test", str(mixed_project), config=config)

        search = HybridSearch(store, embedder)
        results = search.search("setup environment API", project_id="s5-test", file_pattern="docs")

        assert len(results) > 0
        for r in results:
            assert "docs" in r.file_path, f"Expected 'docs' in file_path, got '{r.file_path}'"

        store.close()

    def test_pre_tool_use_blocks_large_file_read(self, mixed_project, db_path):
        """PreToolUse hook blocks Read for large files inside indexed project."""
        config = load_config(mixed_project)
        store = Store(db_path)
        embedder = Embedder()
        indexer = Indexer(store, embedder)
        indexer.index_directory("s5-test", str(mixed_project), config=config)
        store.close()

        large_file = mixed_project / "src" / "big_module.py"
        lines = [f"# line {i}\n" for i in range(350)]
        large_file.write_text("".join(lines))

        hook_input = json.dumps(
            {
                "tool_name": "Read",
                "tool_input": {"file_path": str(large_file.resolve())},
            }
        )

        result = subprocess.run(
            [sys.executable, "-m", "dnomia_knowledge.hooks.pre_tool_use"],
            input=hook_input,
            capture_output=True,
            text=True,
            env={**os.environ, "DNOMIA_KNOWLEDGE_DB": db_path},
        )

        assert result.returncode == 0
        assert result.stdout.strip() != ""
        response = json.loads(result.stdout.strip())
        assert response["decision"] == "block"
        assert "350 lines" in response["reason"]

    def test_pre_tool_use_allows_small_file_read(self, mixed_project, db_path):
        """PreToolUse hook allows Read for small files (< 300 lines)."""
        config = load_config(mixed_project)
        store = Store(db_path)
        embedder = Embedder()
        indexer = Indexer(store, embedder)
        indexer.index_directory("s5-test", str(mixed_project), config=config)
        store.close()

        small_file = mixed_project / "src" / "tiny.py"
        small_file.write_text("x = 1\ny = 2\n")

        hook_input = json.dumps(
            {
                "tool_name": "Read",
                "tool_input": {"file_path": str(small_file.resolve())},
            }
        )

        result = subprocess.run(
            [sys.executable, "-m", "dnomia_knowledge.hooks.pre_tool_use"],
            input=hook_input,
            capture_output=True,
            text=True,
            env={**os.environ, "DNOMIA_KNOWLEDGE_DB": db_path},
        )

        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_pre_tool_use_blocks_grep_in_indexed_project(self, mixed_project, db_path):
        """PreToolUse hook blocks Grep targeting an indexed project directory."""
        config = load_config(mixed_project)
        store = Store(db_path)
        embedder = Embedder()
        indexer = Indexer(store, embedder)
        indexer.index_directory("s5-test", str(mixed_project), config=config)
        store.close()

        hook_input = json.dumps(
            {
                "tool_name": "Grep",
                "tool_input": {
                    "pattern": "def connect",
                    "path": str(mixed_project.resolve()),
                },
            }
        )

        result = subprocess.run(
            [sys.executable, "-m", "dnomia_knowledge.hooks.pre_tool_use"],
            input=hook_input,
            capture_output=True,
            text=True,
            env={**os.environ, "DNOMIA_KNOWLEDGE_DB": db_path},
        )

        assert result.returncode == 0
        assert result.stdout.strip() != ""
        response = json.loads(result.stdout.strip())
        assert response["decision"] == "block"
        assert "search" in response["reason"].lower()
