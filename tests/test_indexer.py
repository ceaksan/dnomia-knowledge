"""Tests for indexing pipeline."""

from __future__ import annotations

import pytest

from dnomia_knowledge.indexer import Indexer
from dnomia_knowledge.registry import (
    CodeConfig,
    ContentConfig,
    GraphConfig,
    IndexingConfig,
    ProjectConfig,
)
from dnomia_knowledge.store import Store


@pytest.fixture
def indexer(db_path, shared_embedder):
    store = Store(db_path)
    return Indexer(store, shared_embedder)


class TestIndexer:
    def test_index_single_file(self, indexer, sample_markdown_file):
        count, chunk_ids = indexer.index_file(
            project_id="test",
            project_path=str(sample_markdown_file.parent),
            file_path=str(sample_markdown_file),
        )
        assert count > 0
        assert len(chunk_ids) == count

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
        assert r1.total_chunks > 0

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
        indexer.index_directory("test-proj", str(tmp_dir))
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


class TestIndexerWithConfig:
    def test_config_scans_code_files(self, tmp_dir, db_path, shared_embedder):
        """With preset='python', .py files indexed as code chunks."""
        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        config = ProjectConfig(
            name="test",
            type="saas",
            content=ContentConfig(extensions=[".md"]),
            code=CodeConfig(preset="python"),
        )
        (tmp_dir / "readme.md").write_text(
            "## Hello\n\n"
            "This is a readme with enough content to be a valid chunk. "
            "It needs to be longer than the minimum chunk size of 200 characters. "
            "So we add more text here to make it work properly for testing."
        )
        (tmp_dir / "app.py").write_text("def hello():\n    return 'hi'\n")
        result = indexer.index_directory("test", str(tmp_dir), config=config)
        assert result.total_files == 2
        assert result.content_chunks > 0
        assert result.code_chunks > 0
        store.close()

    def test_no_config_only_markdown(self, tmp_dir, db_path, shared_embedder):
        """Without config, only .md/.mdx files are indexed (Sprint 1 backward compat)."""
        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        (tmp_dir / "post.md").write_text(
            "## Post Title\n\n"
            "This is a blog post with enough content to pass the minimum chunk size. "
            "We need at least 200 characters for the chunk to be valid. "
            "Adding more text to ensure this works correctly in our test."
        )
        (tmp_dir / "app.py").write_text("def hello(): pass\n")
        result = indexer.index_directory("test", str(tmp_dir))
        assert result.total_files == 1  # only .md
        assert result.code_chunks == 0
        store.close()

    def test_config_content_paths_scoping(self, tmp_dir, db_path, shared_embedder):
        """Config content.paths restricts scanning to specified dirs."""
        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        config = ProjectConfig(
            name="test",
            content=ContentConfig(paths=["docs/"]),
        )
        (tmp_dir / "docs").mkdir()
        (tmp_dir / "docs" / "guide.md").write_text(
            "## Guide\n\n"
            "This is a guide document with sufficient content to meet the minimum "
            "chunk size requirement of 200 characters for the heading-based chunker. "
            "We add extra text here to make it long enough."
        )
        (tmp_dir / "other.md").write_text(
            "## Other\n\n"
            "This file should be ignored because it is outside the docs/ path. "
            "Even though it is a markdown file, the config restricts content to docs/."
        )
        result = indexer.index_directory("test", str(tmp_dir), config=config)
        assert result.total_files == 1  # only docs/guide.md
        store.close()

    def test_config_ignore_patterns(self, tmp_dir, db_path, shared_embedder):
        """Custom ignore patterns from config are respected."""
        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        config = ProjectConfig(
            name="test",
            indexing=IndexingConfig(ignore_patterns=["vendor"]),
        )
        (tmp_dir / "vendor").mkdir()
        (tmp_dir / "vendor" / "lib.md").write_text(
            "## Vendor\n\nThis should be ignored by the vendor ignore pattern."
        )
        (tmp_dir / "readme.md").write_text(
            "## Readme\n\n"
            "This is a readme with enough content to meet the minimum chunk size. "
            "It has more than 200 characters of text content so it will be chunked "
            "properly by the heading-based markdown chunker."
        )
        result = indexer.index_directory("test", str(tmp_dir), config=config)
        assert result.total_files == 1  # only readme.md
        store.close()

    def test_config_max_file_size(self, tmp_dir, db_path, shared_embedder):
        """Files exceeding config max_file_size_kb are skipped."""
        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        config = ProjectConfig(
            name="test",
            indexing=IndexingConfig(max_file_size_kb=1),  # 1KB limit
        )
        big_content = "## Big File\n\n" + "x" * 2000  # > 1KB
        (tmp_dir / "big.md").write_text(big_content)
        (tmp_dir / "small.md").write_text(
            "## Small\n\n"
            "This is a small file that fits within the 1KB limit. "
            "It has enough content to be a valid chunk with more than "
            "200 characters for the minimum chunk size requirement."
        )
        result = indexer.index_directory("test", str(tmp_dir), config=config)
        assert result.total_files == 1  # only small.md
        store.close()


class TestAstChunkerIntegration:
    """Test AST chunker integration through the indexer pipeline."""

    def test_python_file_produces_function_chunks(self, tmp_dir, db_path, shared_embedder):
        """Python code file indexed via AstChunker produces function-level chunks."""
        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        py_content = (
            "def greet(name):\n"
            '    """Return a greeting."""\n'
            "    return f'Hello, {name}!'\n"
            "\n"
            "\n"
            "def farewell(name):\n"
            '    """Return a farewell."""\n'
            "    return f'Goodbye, {name}!'\n"
        )
        config = ProjectConfig(
            name="test",
            type="saas",
            content=ContentConfig(extensions=[".md"]),
            code=CodeConfig(preset="python"),
        )
        (tmp_dir / "funcs.py").write_text(py_content)
        result = indexer.index_directory("test", str(tmp_dir), config=config)
        assert result.code_chunks >= 2

        conn = store._connect()
        rows = conn.execute(
            "SELECT chunk_type FROM chunks WHERE project_id = ? AND file_path = ?",
            ("test", "funcs.py"),
        ).fetchall()
        chunk_types = {r[0] for r in rows}
        assert "function" in chunk_types
        store.close()

    def test_typescript_file_chunks(self, tmp_dir, db_path, shared_embedder):
        """TypeScript code file produces AST-based chunks."""
        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        ts_content = (
            "interface User {\n"
            "  name: string;\n"
            "  age: number;\n"
            "}\n"
            "\n"
            "function getUser(id: number): User {\n"
            "  return { name: 'test', age: 25 };\n"
            "}\n"
        )
        config = ProjectConfig(
            name="test",
            type="saas",
            content=ContentConfig(extensions=[".md"]),
            code=CodeConfig(preset="web"),
        )
        (tmp_dir / "user.ts").write_text(ts_content)
        result = indexer.index_directory("test", str(tmp_dir), config=config)
        assert result.code_chunks >= 1
        store.close()

    def test_syntax_error_still_produces_chunks(self, tmp_dir, db_path, shared_embedder):
        """File with syntax errors falls back to sliding-window, still produces chunks."""
        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        bad_py = (
            "def broken(\n"
            "    # missing closing paren and colon\n"
            "    return 42\n"
            "\n"
            "def also_broken[[\n"
            "    pass\n"
        )
        config = ProjectConfig(
            name="test",
            type="saas",
            content=ContentConfig(extensions=[".md"]),
            code=CodeConfig(preset="python"),
        )
        (tmp_dir / "bad.py").write_text(bad_py)
        result = indexer.index_directory("test", str(tmp_dir), config=config)
        assert result.code_chunks > 0
        store.close()

    def test_graph_enabled_flag_set(self, tmp_dir, db_path, shared_embedder):
        """Config with graph.enabled=True sets graph_enabled in projects table."""
        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        config = ProjectConfig(
            name="test",
            type="content",
            graph=GraphConfig(enabled=True),
        )
        (tmp_dir / "readme.md").write_text(
            "## Hello\n\n"
            "This is a test document with enough content to pass the minimum "
            "chunk size requirement of 200 characters. Adding more text here "
            "to make sure the chunk is valid and gets properly indexed."
        )
        indexer.index_directory("test", str(tmp_dir), config=config)

        conn = store._connect()
        row = conn.execute("SELECT graph_enabled FROM projects WHERE id = ?", ("test",)).fetchone()
        assert row is not None
        assert row[0] == 1
        store.close()

    def test_graph_disabled_by_default(self, tmp_dir, db_path, shared_embedder):
        """Without graph config, graph_enabled defaults to False."""
        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        (tmp_dir / "readme.md").write_text(
            "## Hello\n\n"
            "This is a test document with enough content to pass the minimum "
            "chunk size requirement of 200 characters for proper indexing."
        )
        indexer.index_directory("test", str(tmp_dir))

        conn = store._connect()
        row = conn.execute("SELECT graph_enabled FROM projects WHERE id = ?", ("test",)).fetchone()
        assert row is not None
        assert row[0] == 0
        store.close()
