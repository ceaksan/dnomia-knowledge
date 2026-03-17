"""End-to-end integration tests using real git repos."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dnomia_knowledge.git_sync import GitSync
from dnomia_knowledge.store import Store


def _create_test_repo(tmp_dir: Path) -> Path:
    """Create a real git repo with some commits."""
    repo = tmp_dir / "test-repo"
    repo.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@test.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@test.com",
    }

    subprocess.run(["git", "init"], cwd=str(repo), env=env, capture_output=True)
    subprocess.run(["git", "checkout", "-b", "main"], cwd=str(repo), env=env, capture_output=True)

    # Commit 1: add files
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("print('hello')\n")
    (repo / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), env=env, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feat: initial commit"], cwd=str(repo), env=env, capture_output=True
    )

    # Commit 2: modify file
    (repo / "src" / "main.py").write_text("print('hello')\nprint('world')\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), env=env, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feat: add world"], cwd=str(repo), env=env, capture_output=True
    )

    # Commit 3: add binary
    (repo / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    subprocess.run(["git", "add", "."], cwd=str(repo), env=env, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feat: add logo"], cwd=str(repo), env=env, capture_output=True
    )

    return repo


class TestGitSyncIntegration:
    def test_full_sync(self, tmp_dir):
        repo = _create_test_repo(tmp_dir)
        db_path = str(tmp_dir / "test.db")
        store = Store(db_path)
        store.register_project("test-repo", str(repo), "saas")

        git_sync = GitSync(store)
        result = git_sync.sync("test-repo", str(repo))

        assert result.mode == "full"
        assert result.commits_parsed == 3
        assert result.changes_parsed > 0

        # Verify churn
        rows = store.get_churn("test-repo", days=90, limit=20)
        assert len(rows) > 0
        file_paths = [r["file_path"] for r in rows]
        assert "src/main.py" in file_paths

        store.close()

    def test_incremental_sync(self, tmp_dir):
        repo = _create_test_repo(tmp_dir)
        db_path = str(tmp_dir / "test.db")
        store = Store(db_path)
        store.register_project("test-repo", str(repo), "saas")

        git_sync = GitSync(store)

        # First sync
        r1 = git_sync.sync("test-repo", str(repo))
        assert r1.mode == "full"

        # Add another commit
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@test.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@test.com",
        }
        (repo / "src" / "util.py").write_text("def helper(): pass\n")
        subprocess.run(["git", "add", "."], cwd=str(repo), env=env, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "feat: add util"], cwd=str(repo), env=env, capture_output=True
        )

        # Incremental sync
        r2 = git_sync.sync("test-repo", str(repo))
        assert r2.mode == "incremental"
        assert r2.commits_parsed == 1

        store.close()

    def test_binary_file_handling(self, tmp_dir):
        repo = _create_test_repo(tmp_dir)
        db_path = str(tmp_dir / "test.db")
        store = Store(db_path)
        store.register_project("test-repo", str(repo), "saas")

        git_sync = GitSync(store)
        git_sync.sync("test-repo", str(repo))

        # Check binary file has NULL insertions/deletions
        conn = store._connect()
        row = conn.execute(
            "SELECT * FROM git_file_changes"
            " WHERE file_path = 'logo.png' AND project_id = 'test-repo'"
        ).fetchone()
        if row:
            row_dict = dict(row)
            assert row_dict["is_binary"] == 1
            assert row_dict["insertions"] is None
            assert row_dict["deletions"] is None

        store.close()

    def test_force_resync(self, tmp_dir):
        repo = _create_test_repo(tmp_dir)
        db_path = str(tmp_dir / "test.db")
        store = Store(db_path)
        store.register_project("test-repo", str(repo), "saas")

        git_sync = GitSync(store)
        git_sync.sync("test-repo", str(repo))
        r2 = git_sync.sync("test-repo", str(repo), force=True)
        assert r2.mode == "full"

        store.close()
