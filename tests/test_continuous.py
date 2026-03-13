"""Tests for continuous indexing features."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from dnomia_knowledge.store import Store


class TestStoreLastIndexed:
    def test_update_project_last_indexed(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        store.update_project_last_indexed("test")
        proj = store.get_project("test")
        assert proj["last_indexed"] is not None
        store.close()

    def test_update_project_last_indexed_with_commit(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        store.update_project_last_indexed("test", commit_hash="abc123")
        proj = store.get_project("test")
        assert proj["last_indexed"] is not None
        assert proj["last_indexed_commit"] == "abc123"
        store.close()

    def test_last_indexed_commit_column_exists(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        proj = store.get_project("test")
        assert "last_indexed_commit" in proj
        store.close()


class TestIndexerLastIndexed:
    def test_index_directory_updates_last_indexed(
        self, db_path, shared_embedder, tmp_dir, sample_markdown
    ):
        from dnomia_knowledge.indexer import Indexer
        from dnomia_knowledge.store import Store

        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        md = tmp_dir / "post.md"
        md.write_text(sample_markdown)

        indexer.index_directory("test-proj", str(tmp_dir))
        proj = store.get_project("test-proj")
        assert proj["last_indexed"] is not None
        store.close()

    def test_index_directory_updates_git_commit(
        self, db_path, shared_embedder, tmp_dir, sample_markdown
    ):
        from dnomia_knowledge.indexer import Indexer
        from dnomia_knowledge.store import Store

        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        # Create a git repo with a commit
        subprocess.run(["git", "init"], cwd=str(tmp_dir), capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"], cwd=str(tmp_dir), capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], cwd=str(tmp_dir), capture_output=True
        )
        md = tmp_dir / "post.md"
        md.write_text(sample_markdown)
        subprocess.run(["git", "add", "."], cwd=str(tmp_dir), capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_dir), capture_output=True)

        indexer.index_directory("test-proj", str(tmp_dir))
        proj = store.get_project("test-proj")
        assert proj["last_indexed_commit"] is not None
        assert len(proj["last_indexed_commit"]) == 40  # Full SHA
        store.close()
