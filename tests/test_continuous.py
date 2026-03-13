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


class TestIndexAll:
    def test_index_all_indexes_registered_projects(
        self, db_path, shared_embedder, tmp_dir, sample_markdown
    ):
        from dnomia_knowledge.indexer import Indexer
        from dnomia_knowledge.store import Store

        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        # Create two project dirs
        proj1 = tmp_dir / "proj1"
        proj1.mkdir()
        (proj1 / "post.md").write_text(sample_markdown)

        proj2 = tmp_dir / "proj2"
        proj2.mkdir()
        (proj2 / "doc.md").write_text("## Doc\n\nSome documentation content here.")

        # Register them via initial index
        indexer.index_directory("proj1", str(proj1))
        indexer.index_directory("proj2", str(proj2))

        # Modify a file
        (proj1 / "post.md").write_text("## Updated\n\nUpdated content for testing incremental.")

        results = indexer.index_all(lock=False)
        assert len(results) == 2
        store.close()

    def test_index_all_changed_only(self, db_path, shared_embedder, tmp_dir, sample_markdown):
        from dnomia_knowledge.indexer import Indexer
        from dnomia_knowledge.store import Store

        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        proj1 = tmp_dir / "proj1"
        proj1.mkdir()
        (proj1 / "post.md").write_text(sample_markdown)

        # Init as git repo with commit
        subprocess.run(["git", "init"], cwd=str(proj1), capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"], cwd=str(proj1), capture_output=True
        )
        subprocess.run(["git", "config", "user.name", "T"], cwd=str(proj1), capture_output=True)
        subprocess.run(["git", "add", "."], cwd=str(proj1), capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(proj1), capture_output=True)

        indexer.index_directory("proj1", str(proj1))

        # No changes since last index, --changed should skip
        results = indexer.index_all(changed_only=True, lock=False)
        assert len(results) == 0  # Skipped, no changes
        store.close()

    def test_index_all_skips_missing_project(
        self, db_path, shared_embedder, tmp_dir, sample_markdown
    ):
        from dnomia_knowledge.indexer import Indexer
        from dnomia_knowledge.store import Store
        import shutil

        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        proj = tmp_dir / "proj"
        proj.mkdir()
        (proj / "post.md").write_text(sample_markdown)
        indexer.index_directory("proj", str(proj))

        # Delete the project directory
        shutil.rmtree(proj)

        # Should not raise, just log warning and skip
        results = indexer.index_all(lock=False)
        assert len(results) == 0
        store.close()


class TestCLIParser:
    def test_index_all_command(self):
        from dnomia_knowledge.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["index-all"])
        assert args.command == "index-all"
        assert args.changed is False

    def test_index_all_changed(self):
        from dnomia_knowledge.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["index-all", "--changed"])
        assert args.changed is True
