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


class TestInstallHooks:
    def test_install_hooks_creates_hook(self, db_path, tmp_dir, sample_markdown):
        from dnomia_knowledge.store import Store

        store = Store(db_path)

        # Create a git project
        proj = tmp_dir / "proj"
        proj.mkdir()
        (proj / "post.md").write_text(sample_markdown)
        subprocess.run(["git", "init"], cwd=str(proj), capture_output=True)

        store.register_project("proj", str(proj), "content")

        from dnomia_knowledge.cli import _install_hooks

        installed = _install_hooks(store, bin_path="/usr/local/bin/dnomia-knowledge")

        hook_path = proj / ".git" / "hooks" / "post-commit"
        assert hook_path.exists()
        content = hook_path.read_text()
        assert "dnomia-knowledge-start" in content
        assert "dnomia-knowledge-end" in content
        assert installed == 1
        store.close()

    def test_install_hooks_idempotent(self, db_path, tmp_dir, sample_markdown):
        from dnomia_knowledge.store import Store

        store = Store(db_path)

        proj = tmp_dir / "proj"
        proj.mkdir()
        (proj / "post.md").write_text(sample_markdown)
        subprocess.run(["git", "init"], cwd=str(proj), capture_output=True)
        store.register_project("proj", str(proj), "content")

        from dnomia_knowledge.cli import _install_hooks

        _install_hooks(store, bin_path="/usr/local/bin/dnomia-knowledge")
        _install_hooks(store, bin_path="/usr/local/bin/dnomia-knowledge")

        hook_path = proj / ".git" / "hooks" / "post-commit"
        content = hook_path.read_text()
        # Should appear only once
        assert content.count("dnomia-knowledge-start") == 1
        store.close()

    def test_install_hooks_chains_existing(self, db_path, tmp_dir, sample_markdown):
        from dnomia_knowledge.store import Store

        store = Store(db_path)

        proj = tmp_dir / "proj"
        proj.mkdir()
        (proj / "post.md").write_text(sample_markdown)
        subprocess.run(["git", "init"], cwd=str(proj), capture_output=True)

        # Pre-existing hook
        hook_path = proj / ".git" / "hooks" / "post-commit"
        hook_path.write_text("#!/bin/sh\necho 'existing hook'\n")
        hook_path.chmod(0o755)

        store.register_project("proj", str(proj), "content")

        from dnomia_knowledge.cli import _install_hooks

        _install_hooks(store, bin_path="/usr/local/bin/dnomia-knowledge")

        content = hook_path.read_text()
        assert "existing hook" in content
        assert "dnomia-knowledge-start" in content
        store.close()

    def test_uninstall_hooks(self, db_path, tmp_dir, sample_markdown):
        from dnomia_knowledge.store import Store

        store = Store(db_path)

        proj = tmp_dir / "proj"
        proj.mkdir()
        (proj / "post.md").write_text(sample_markdown)
        subprocess.run(["git", "init"], cwd=str(proj), capture_output=True)
        store.register_project("proj", str(proj), "content")

        from dnomia_knowledge.cli import _install_hooks, _uninstall_hooks

        _install_hooks(store, bin_path="/usr/local/bin/dnomia-knowledge")
        removed = _uninstall_hooks(store)

        hook_path = proj / ".git" / "hooks" / "post-commit"
        if hook_path.exists():
            content = hook_path.read_text()
            assert "dnomia-knowledge-start" not in content
        assert removed == 1
        store.close()

    def test_install_hooks_skips_non_git(self, db_path, tmp_dir):
        from dnomia_knowledge.store import Store

        store = Store(db_path)
        proj = tmp_dir / "proj"
        proj.mkdir()
        store.register_project("proj", str(proj), "content")

        from dnomia_knowledge.cli import _install_hooks

        installed = _install_hooks(store, bin_path="/usr/local/bin/dnomia-knowledge")
        assert installed == 0
        store.close()


class TestInstallLaunchd:
    def test_generate_plist(self):
        from dnomia_knowledge.cli import _generate_plist

        plist = _generate_plist("/path/to/bin/dnomia-knowledge")
        assert "com.dnomia-knowledge.index" in plist
        assert "/path/to/bin/dnomia-knowledge" in plist
        assert "<integer>300</integer>" in plist
        assert "index-all" in plist
        assert "--changed" in plist

    def test_install_launchd_creates_plist(self, tmp_dir):
        from dnomia_knowledge.cli import _generate_plist

        plist_content = _generate_plist("/path/to/bin/dnomia-knowledge")
        plist_path = tmp_dir / "com.dnomia-knowledge.index.plist"
        plist_path.write_text(plist_content)
        assert plist_path.exists()
        assert "<?xml" in plist_path.read_text()

    def test_cli_parser_install_launchd(self):
        from dnomia_knowledge.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["install-launchd"])
        assert args.command == "install-launchd"
        assert args.uninstall is False

    def test_cli_parser_uninstall_launchd(self):
        from dnomia_knowledge.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["install-launchd", "--uninstall"])
        assert args.uninstall is True


class TestIntegration:
    def test_full_flow_index_commit_reindex(
        self, db_path, shared_embedder, tmp_dir, sample_markdown
    ):
        """Full flow: index -> commit -> index-all --changed detects change."""
        from dnomia_knowledge.indexer import Indexer
        from dnomia_knowledge.store import Store

        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        proj = tmp_dir / "proj"
        proj.mkdir()
        (proj / "post.md").write_text(sample_markdown)

        subprocess.run(["git", "init"], cwd=str(proj), capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"], cwd=str(proj), capture_output=True
        )
        subprocess.run(["git", "config", "user.name", "T"], cwd=str(proj), capture_output=True)
        subprocess.run(["git", "add", "."], cwd=str(proj), capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(proj), capture_output=True)

        # Initial index
        indexer.index_directory("proj", str(proj))
        proj_data = store.get_project("proj")
        assert proj_data["last_indexed"] is not None
        assert proj_data["last_indexed_commit"] is not None
        initial_commit = proj_data["last_indexed_commit"]

        # No changes: index-all --changed should skip
        results = indexer.index_all(changed_only=True, lock=False)
        assert len(results) == 0

        # Make a change and commit
        (proj / "post.md").write_text("## Changed\n\nNew content for the post.")
        subprocess.run(["git", "add", "."], cwd=str(proj), capture_output=True)
        subprocess.run(["git", "commit", "-m", "update"], cwd=str(proj), capture_output=True)

        # Now index-all --changed should pick it up
        results = indexer.index_all(changed_only=True, lock=False)
        assert len(results) == 1
        assert results[0].project_id == "proj"

        # Verify commit hash updated
        proj_data = store.get_project("proj")
        assert proj_data["last_indexed_commit"] != initial_commit
        store.close()
