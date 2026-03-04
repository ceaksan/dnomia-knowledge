"""Integration tests for Sprint 2 and Sprint 3."""

from __future__ import annotations

import json
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
