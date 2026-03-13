# Continuous Indexing Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatic index updates via git post-commit hooks and launchd periodic job, with no new dependencies.

**Architecture:** File lock prevents concurrent index runs. Git hooks handle commit-time indexing. launchd periodic job catches non-git changes (vault edits). `index-all --changed` uses git HEAD comparison for fast skip detection.

**Tech Stack:** Python 3.11+, SQLite, fcntl (stdlib), subprocess (stdlib), argparse (stdlib)

**Spec:** `docs/specs/2026-03-13-continuous-indexing-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `src/dnomia_knowledge/lock.py` | File lock utility (fcntl + mkdir) | Create |
| `src/dnomia_knowledge/cli.py` | CLI commands | Modify (add index-all, install-hooks, install-launchd) |
| `src/dnomia_knowledge/indexer.py` | Indexing pipeline | Modify (add index_all, update last_indexed) |
| `src/dnomia_knowledge/store.py` | SQLite store | Modify (add update_project_last_indexed, last_indexed_commit column) |
| `tests/test_lock.py` | Lock tests | Create |
| `tests/test_continuous.py` | index-all, install-hooks tests | Create |

---

## Chunk 1: File Lock + Store Updates

### Task 1: File lock utility

**Files:**
- Create: `src/dnomia_knowledge/lock.py`
- Test: `tests/test_lock.py`

- [ ] **Step 1: Write failing tests for lock module**

```python
# tests/test_lock.py
"""Tests for file lock utility."""

from __future__ import annotations

import os
import tempfile

import pytest

from dnomia_knowledge.lock import IndexLock


class TestIndexLock:
    def test_acquire_and_release(self, tmp_dir):
        lock_path = str(tmp_dir / "test.lock")
        lock = IndexLock(lock_path)
        assert lock.acquire() is True
        lock.release()

    def test_non_blocking_fails_when_held(self, tmp_dir):
        lock_path = str(tmp_dir / "test.lock")
        lock1 = IndexLock(lock_path)
        lock2 = IndexLock(lock_path)
        assert lock1.acquire() is True
        assert lock2.acquire() is False
        lock1.release()

    def test_acquire_after_release(self, tmp_dir):
        lock_path = str(tmp_dir / "test.lock")
        lock1 = IndexLock(lock_path)
        lock2 = IndexLock(lock_path)
        assert lock1.acquire() is True
        lock1.release()
        assert lock2.acquire() is True
        lock2.release()

    def test_context_manager(self, tmp_dir):
        lock_path = str(tmp_dir / "test.lock")
        with IndexLock(lock_path) as acquired:
            assert acquired is True

    def test_context_manager_skip_when_held(self, tmp_dir):
        lock_path = str(tmp_dir / "test.lock")
        lock1 = IndexLock(lock_path)
        assert lock1.acquire() is True
        with IndexLock(lock_path) as acquired:
            assert acquired is False
        lock1.release()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_lock.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement lock module**

```python
# src/dnomia_knowledge/lock.py
"""File lock utility for preventing concurrent index runs."""

from __future__ import annotations

import fcntl
import os

LOCK_PATH = "/tmp/dnomia-knowledge.lockfile"


class IndexLock:
    """Non-blocking file lock using fcntl."""

    def __init__(self, path: str = LOCK_PATH):
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Try to acquire lock. Returns True if acquired, False if held by another process."""
        try:
            self._fd = os.open(self.path, os.O_CREAT | os.O_WRONLY)
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            return False

    def release(self) -> None:
        """Release the lock."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *args) -> None:
        self.release()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_lock.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge
git add src/dnomia_knowledge/lock.py tests/test_lock.py
git commit -m "feat: add file lock utility for concurrent index prevention"
```

---

### Task 2: Store - add last_indexed_commit and update_project_last_indexed

**Files:**
- Modify: `src/dnomia_knowledge/store.py:180-199` (register_project) and add new method
- Test: `tests/test_continuous.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_continuous.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_continuous.py::TestStoreLastIndexed -v`
Expected: FAIL (missing column and method)

- [ ] **Step 3: Add last_indexed_commit column to schema and update_project_last_indexed method**

In `store.py`, add `last_indexed_commit TEXT` column to `projects` table in `_TABLES_SQL`:

```python
# In _TABLES_SQL, change projects table to:
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL,
    graph_enabled INTEGER DEFAULT 0,
    last_indexed TEXT,
    last_indexed_commit TEXT,
    config_hash TEXT
);
```

Add migration in `_init_db` (after `_TABLES_SQL` executescript):

```python
# Add column if not exists (migration for existing DBs)
try:
    conn.execute("ALTER TABLE projects ADD COLUMN last_indexed_commit TEXT")
except sqlite3.OperationalError:
    pass  # Column already exists
```

Add new method to `Store` class after `list_projects`:

```python
def update_project_last_indexed(
    self, project_id: str, commit_hash: str | None = None
) -> None:
    """Update last_indexed timestamp and optionally commit hash."""
    conn = self._connect()
    if commit_hash:
        conn.execute(
            "UPDATE projects SET last_indexed = datetime('now'), last_indexed_commit = ? WHERE id = ?",
            (commit_hash, project_id),
        )
    else:
        conn.execute(
            "UPDATE projects SET last_indexed = datetime('now') WHERE id = ?",
            (project_id,),
        )
    conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_continuous.py::TestStoreLastIndexed -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run existing store tests to verify no regressions**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_store.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge
git add src/dnomia_knowledge/store.py tests/test_continuous.py
git commit -m "feat: add last_indexed_commit tracking to projects table"
```

---

### Task 3: Update indexer to set last_indexed after success

**Files:**
- Modify: `src/dnomia_knowledge/indexer.py:282-293` (end of index_directory)

- [ ] **Step 1: Write failing test**

Add to `tests/test_continuous.py`:

```python
class TestIndexerLastIndexed:
    def test_index_directory_updates_last_indexed(self, db_path, shared_embedder, tmp_dir, sample_markdown):
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

    def test_index_directory_updates_git_commit(self, db_path, shared_embedder, tmp_dir, sample_markdown):
        from dnomia_knowledge.indexer import Indexer
        from dnomia_knowledge.store import Store

        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        # Create a git repo with a commit
        subprocess.run(["git", "init"], cwd=str(tmp_dir), capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(tmp_dir), capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_dir), capture_output=True)
        md = tmp_dir / "post.md"
        md.write_text(sample_markdown)
        subprocess.run(["git", "add", "."], cwd=str(tmp_dir), capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_dir), capture_output=True)

        indexer.index_directory("test-proj", str(tmp_dir))
        proj = store.get_project("test-proj")
        assert proj["last_indexed_commit"] is not None
        assert len(proj["last_indexed_commit"]) == 40  # Full SHA
        store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_continuous.py::TestIndexerLastIndexed -v`
Expected: FAIL (last_indexed not being set)

- [ ] **Step 3: Update index_directory to set last_indexed**

At the end of `index_directory` in `indexer.py`, before the `return` statement (after line 283, before line 284):

```python
        # Update last_indexed
        commit_hash = _get_git_head(project_path)
        self.store.update_project_last_indexed(project_id, commit_hash=commit_hash)
```

Add helper function at module level (after `_compute_file_hash`):

```python
def _get_git_head(directory: str) -> str | None:
    """Get current git HEAD commit hash, or None if not a git repo."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_continuous.py::TestIndexerLastIndexed -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Run full indexer tests**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_indexer.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge
git add src/dnomia_knowledge/indexer.py tests/test_continuous.py
git commit -m "feat: update last_indexed and git commit hash after successful indexing"
```

---

## Chunk 2: CLI Commands

### Task 4: index-all command

**Files:**
- Modify: `src/dnomia_knowledge/cli.py`
- Modify: `src/dnomia_knowledge/indexer.py` (add index_all method)
- Test: `tests/test_continuous.py`

- [ ] **Step 1: Write failing tests for index_all**

Add to `tests/test_continuous.py`:

```python
class TestIndexAll:
    def test_index_all_indexes_registered_projects(self, db_path, shared_embedder, tmp_dir, sample_markdown):
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
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(proj1), capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=str(proj1), capture_output=True)
        subprocess.run(["git", "add", "."], cwd=str(proj1), capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(proj1), capture_output=True)

        indexer.index_directory("proj1", str(proj1))

        # No changes since last index, --changed should skip
        results = indexer.index_all(changed_only=True, lock=False)
        assert len(results) == 0  # Skipped, no changes
        store.close()

    def test_index_all_skips_missing_project(self, db_path, shared_embedder, tmp_dir, sample_markdown):
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_continuous.py::TestIndexAll -v`
Expected: FAIL (index_all method doesn't exist)

- [ ] **Step 3: Implement index_all in Indexer**

Add to `Indexer` class in `indexer.py`:

```python
    def index_all(
        self,
        changed_only: bool = False,
        lock: bool = True,
    ) -> list[IndexResult]:
        """Index all registered projects. Returns list of IndexResult for indexed projects."""
        if lock:
            from dnomia_knowledge.lock import IndexLock
            _lock = IndexLock()
            if not _lock.acquire():
                logger.info("Another index process is running, skipping.")
                return []
        else:
            _lock = None

        try:
            projects = self.store.list_projects()
            results = []

            for proj in projects:
                project_path = proj["path"]
                project_id = proj["id"]

                if not os.path.isdir(project_path):
                    logger.warning("Project directory missing, skipping: %s (%s)", project_id, project_path)
                    continue

                if changed_only and not self._project_has_changes(proj):
                    logger.debug("No changes detected, skipping: %s", project_id)
                    continue

                try:
                    from dnomia_knowledge.registry import load_config, default_config
                    config = load_config(project_path)
                    if config is None:
                        config = default_config(project_path)

                    result = self.index_directory(
                        project_id=project_id,
                        directory=project_path,
                        incremental=True,
                        config=config,
                    )
                    results.append(result)
                    logger.info(
                        "Indexed %s: %d files, %d chunks, %.1fs",
                        project_id, result.indexed_files, result.total_chunks, result.duration_seconds,
                    )
                except Exception as e:
                    logger.warning("Failed to index %s: %s", project_id, e)

            return results
        finally:
            if _lock:
                _lock.release()

    def _project_has_changes(self, proj: dict) -> bool:
        """Check if project has changes since last index."""
        project_path = proj["path"]
        stored_commit = proj.get("last_indexed_commit")

        # Try git HEAD comparison first
        current_commit = _get_git_head(project_path)
        if current_commit and stored_commit:
            return current_commit != stored_commit
        if current_commit and not stored_commit:
            return True  # Never indexed with commit tracking

        # Fallback: check file mtimes against last_indexed
        last_indexed = proj.get("last_indexed")
        if not last_indexed:
            return True  # Never indexed

        from datetime import datetime
        try:
            last_dt = datetime.fromisoformat(last_indexed)
            last_ts = last_dt.timestamp()
        except (ValueError, TypeError):
            return True

        # Recursive walk to catch nested changes (vault subdirs, etc.)
        for root, _dirs, files in os.walk(project_path):
            for fname in files:
                try:
                    fpath = os.path.join(root, fname)
                    if os.stat(fpath).st_mtime > last_ts:
                        return True
                except OSError:
                    continue

        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_continuous.py::TestIndexAll -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Add CLI parser entry for index-all**

In `cli.py`, add `cmd_index_all` function and parser entry:

```python
def cmd_index_all(args: argparse.Namespace) -> None:
    """Index all registered projects."""
    from dnomia_knowledge.embedder import Embedder
    from dnomia_knowledge.indexer import Indexer
    from dnomia_knowledge.store import Store

    store = Store(_get_db_path())
    embedder = Embedder()
    indexer = Indexer(store, embedder)

    results = indexer.index_all(changed_only=args.changed)

    if not results:
        if args.changed:
            console.print("[dim]No projects with changes.[/dim]")
        else:
            console.print("[dim]No projects registered.[/dim]")
    else:
        for r in results:
            console.print(
                f"  [green]{r.project_id}[/green]: "
                f"{r.indexed_files} files, {r.total_chunks} chunks, {r.duration_seconds}s"
            )

    store.close()
```

In `build_parser`, add after the `index` subparser:

```python
    # index-all
    p_index_all = subparsers.add_parser("index-all", help="Index all registered projects")
    p_index_all.add_argument("--changed", action="store_true", help="Only index projects with changes")
    p_index_all.set_defaults(func=cmd_index_all)
```

- [ ] **Step 6: Add CLI parser test**

Add to `tests/test_continuous.py`:

```python
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
```

- [ ] **Step 7: Run all continuous tests**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_continuous.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge
git add src/dnomia_knowledge/indexer.py src/dnomia_knowledge/cli.py tests/test_continuous.py
git commit -m "feat: add index-all command for batch project indexing"
```

---

### Task 5: install-hooks command

**Files:**
- Modify: `src/dnomia_knowledge/cli.py`
- Test: `tests/test_continuous.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_continuous.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_continuous.py::TestInstallHooks -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement _install_hooks and _uninstall_hooks**

Add `from pathlib import Path` to `cli.py` imports (top of file, after existing imports).

Then add:

```python
HOOK_START_MARKER = "# --- dnomia-knowledge-start ---"
HOOK_END_MARKER = "# --- dnomia-knowledge-end ---"


def _get_hook_block(bin_path: str) -> str:
    return (
        f'{HOOK_START_MARKER}\n'
        f'DNOMIA_BIN="{bin_path}"\n'
        f'if [ -x "$DNOMIA_BIN" ]; then\n'
        f'  (\n'
        f'    mkdir /tmp/dnomia-knowledge.lock 2>/dev/null || exit 0\n'
        f'    trap \'rmdir /tmp/dnomia-knowledge.lock\' EXIT\n'
        f'    "$DNOMIA_BIN" index "$(git rev-parse --show-toplevel)" >/dev/null 2>&1\n'
        f'  ) &\n'
        f'fi\n'
        f'{HOOK_END_MARKER}\n'
    )


def _install_hooks(store, bin_path: str | None = None) -> int:
    """Install post-commit hooks to all registered git projects. Returns count installed."""
    if bin_path is None:
        import shutil
        bin_path = shutil.which("dnomia-knowledge") or sys.executable.replace("python", "dnomia-knowledge")

    projects = store.list_projects()
    installed = 0

    for proj in projects:
        project_path = Path(proj["path"])
        git_dir = project_path / ".git"
        if not git_dir.is_dir():
            continue

        hooks_dir = git_dir / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        hook_path = hooks_dir / "post-commit"

        hook_block = _get_hook_block(bin_path)

        if hook_path.exists():
            existing = hook_path.read_text()
            if HOOK_START_MARKER in existing:
                continue  # Already installed
            new_content = existing.rstrip("\n") + "\n\n" + hook_block
        else:
            new_content = "#!/bin/sh\n\n" + hook_block

        hook_path.write_text(new_content)
        hook_path.chmod(0o755)
        installed += 1

    return installed


def _uninstall_hooks(store) -> int:
    """Remove dnomia-knowledge blocks from all post-commit hooks. Returns count removed."""
    projects = store.list_projects()
    removed = 0

    for proj in projects:
        project_path = Path(proj["path"])
        hook_path = project_path / ".git" / "hooks" / "post-commit"
        if not hook_path.exists():
            continue

        content = hook_path.read_text()
        if HOOK_START_MARKER not in content:
            continue

        # Remove the block
        lines = content.split("\n")
        new_lines = []
        skip = False
        for line in lines:
            if line.strip() == HOOK_START_MARKER.strip():
                skip = True
                continue
            if line.strip() == HOOK_END_MARKER.strip():
                skip = False
                continue
            if not skip:
                new_lines.append(line)

        new_content = "\n".join(new_lines).strip()
        if new_content and new_content != "#!/bin/sh":
            hook_path.write_text(new_content + "\n")
        else:
            hook_path.unlink()
        removed += 1

    return removed


def cmd_install_hooks(args: argparse.Namespace) -> None:
    """Install or uninstall git post-commit hooks."""
    from dnomia_knowledge.store import Store

    store = Store(_get_db_path())

    if args.uninstall:
        count = _uninstall_hooks(store)
        console.print(f"[green]Removed hooks from {count} projects.[/green]")
    else:
        count = _install_hooks(store)
        console.print(f"[green]Installed hooks in {count} projects.[/green]")

    store.close()
```

In `build_parser`, add:

```python
    # install-hooks
    p_hooks = subparsers.add_parser("install-hooks", help="Install git post-commit hooks")
    p_hooks.add_argument("--uninstall", action="store_true", help="Remove hooks instead")
    p_hooks.set_defaults(func=cmd_install_hooks)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_continuous.py::TestInstallHooks -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge
git add src/dnomia_knowledge/cli.py tests/test_continuous.py
git commit -m "feat: add install-hooks command for git post-commit hook management"
```

---

### Task 6: install-launchd command

**Files:**
- Modify: `src/dnomia_knowledge/cli.py`
- Test: `tests/test_continuous.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_continuous.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_continuous.py::TestInstallLaunchd -v`
Expected: FAIL

- [ ] **Step 3: Implement _generate_plist and cmd_install_launchd**

Add to `cli.py`:

```python
PLIST_LABEL = "com.dnomia-knowledge.index"


def _generate_plist(bin_path: str) -> str:
    """Generate launchd plist XML."""
    home = os.path.expanduser("~")
    log_dir = os.path.join(home, ".local", "share", "dnomia-knowledge")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{bin_path}</string>
        <string>index-all</string>
        <string>--changed</string>
    </array>
    <key>StartInterval</key>
    <integer>300</integer>
    <key>StandardOutPath</key>
    <string>{log_dir}/index.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/index-error.log</string>
</dict>
</plist>
"""


def cmd_install_launchd(args: argparse.Namespace) -> None:
    """Install or uninstall launchd periodic job."""
    import shutil
    import subprocess as sp

    home = os.path.expanduser("~")
    plist_path = os.path.join(home, "Library", "LaunchAgents", f"{PLIST_LABEL}.plist")
    log_dir = os.path.join(home, ".local", "share", "dnomia-knowledge")

    if args.uninstall:
        if os.path.exists(plist_path):
            sp.run(["launchctl", "unload", plist_path], capture_output=True)
            os.remove(plist_path)
            console.print(f"[green]Unloaded and removed {plist_path}[/green]")
        else:
            console.print("[dim]No plist found.[/dim]")
        return

    bin_path = shutil.which("dnomia-knowledge")
    if not bin_path:
        # Try venv
        venv_bin = os.path.join(os.path.dirname(sys.executable), "dnomia-knowledge")
        if os.path.exists(venv_bin):
            bin_path = venv_bin
        else:
            console.print("[red]Could not find dnomia-knowledge binary.[/red]")
            sys.exit(1)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(plist_path), exist_ok=True)

    plist_content = _generate_plist(bin_path)

    # Unload if existing
    if os.path.exists(plist_path):
        sp.run(["launchctl", "unload", plist_path], capture_output=True)

    with open(plist_path, "w") as f:
        f.write(plist_content)

    sp.run(["launchctl", "load", plist_path], capture_output=True)
    console.print(f"[green]Installed and loaded {plist_path}[/green]")
    console.print(f"[dim]Logs: {log_dir}/index.log[/dim]")
```

In `build_parser`, add:

```python
    # install-launchd
    p_launchd = subparsers.add_parser("install-launchd", help="Install launchd periodic indexing job")
    p_launchd.add_argument("--uninstall", action="store_true", help="Remove launchd job")
    p_launchd.set_defaults(func=cmd_install_launchd)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_continuous.py::TestInstallLaunchd -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run ALL tests**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/ -v --ignore=tests/test_integration.py`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge
git add src/dnomia_knowledge/cli.py tests/test_continuous.py
git commit -m "feat: add install-launchd command for periodic indexing"
```

---

## Chunk 3: Integration + Final Verification

### Task 7: End-to-end integration test

**Files:**
- Test: `tests/test_continuous.py`

- [ ] **Step 1: Write integration test**

Add to `tests/test_continuous.py`:

```python
class TestIntegration:
    def test_full_flow_index_commit_reindex(self, db_path, shared_embedder, tmp_dir, sample_markdown):
        """Full flow: index -> commit -> index-all --changed detects change."""
        from dnomia_knowledge.indexer import Indexer
        from dnomia_knowledge.store import Store

        store = Store(db_path)
        indexer = Indexer(store, shared_embedder)

        proj = tmp_dir / "proj"
        proj.mkdir()
        (proj / "post.md").write_text(sample_markdown)

        subprocess.run(["git", "init"], cwd=str(proj), capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(proj), capture_output=True)
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
```

- [ ] **Step 2: Run integration test**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/test_continuous.py::TestIntegration -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge
git add tests/test_continuous.py
git commit -m "test: add integration test for continuous indexing flow"
```

---

### Task 8: Setup and verify

- [ ] **Step 1: Install hooks on all indexed projects**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/dnomia-knowledge install-hooks`

- [ ] **Step 2: Install launchd periodic job**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/dnomia-knowledge install-launchd`

- [ ] **Step 3: Verify launchd is loaded**

Run: `launchctl list | grep dnomia-knowledge`
Expected: Shows `com.dnomia-knowledge.index`

- [ ] **Step 4: Test manual index-all**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/dnomia-knowledge index-all --changed`

- [ ] **Step 5: Final commit with spec status update**

Update spec status from Draft to Implemented. Commit.
