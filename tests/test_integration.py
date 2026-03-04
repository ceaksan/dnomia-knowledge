"""Integration tests for Sprint 2: config-driven indexing pipeline."""

from __future__ import annotations

import pytest

from dnomia_knowledge.embedder import Embedder
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
