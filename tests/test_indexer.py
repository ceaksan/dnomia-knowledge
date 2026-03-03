"""Tests for indexing pipeline."""

from __future__ import annotations

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
