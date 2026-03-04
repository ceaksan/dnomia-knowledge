"""Tests for PreToolUse and PostToolUse hook scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from dnomia_knowledge.store import Store

HOOK_SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "dnomia_knowledge",
    "hooks",
    "post_tool_use.py",
)


def _run_hook(stdin_data: str, db_path: str) -> subprocess.CompletedProcess:
    """Run the hook script as a subprocess with given stdin and DB path."""
    env = os.environ.copy()
    env["DNOMIA_KNOWLEDGE_DB"] = db_path
    return subprocess.run(
        [sys.executable, "-m", "dnomia_knowledge.hooks.post_tool_use"],
        input=stdin_data,
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )


def _setup_project(db_path: str, project_path: str, file_rel: str) -> list[int]:
    """Register a project and insert chunks for a file. Returns chunk IDs."""
    store = Store(db_path)
    store.register_project("test-proj", project_path, "content")
    chunk_ids = store.insert_chunks(
        "test-proj",
        [
            {"file_path": file_rel, "content": "chunk one"},
            {"file_path": file_rel, "content": "chunk two"},
        ],
    )
    store.close()
    return chunk_ids


def _get_interactions(db_path: str) -> list[dict]:
    """Read all chunk_interactions rows from DB."""
    store = Store(db_path)
    conn = store._connect()
    rows = conn.execute(
        "SELECT chunk_id, interaction, source_tool FROM chunk_interactions"
    ).fetchall()
    store.close()
    return [dict(r) for r in rows]


class TestPostToolUseHook:
    def test_read_tool_logs_interaction(self, db_path, tmp_dir):
        project_path = str(tmp_dir / "myproject")
        os.makedirs(project_path, exist_ok=True)
        file_rel = "src/app.py"
        file_abs = os.path.join(project_path, file_rel)
        os.makedirs(os.path.dirname(file_abs), exist_ok=True)
        with open(file_abs, "w") as f:
            f.write("print('hello')")

        chunk_ids = _setup_project(db_path, project_path, file_rel)

        stdin_data = json.dumps({"tool_name": "Read", "tool_input": {"file_path": file_abs}})
        result = _run_hook(stdin_data, db_path)
        assert result.returncode == 0

        interactions = _get_interactions(db_path)
        assert len(interactions) == 2
        for row in interactions:
            assert row["interaction"] == "read"
            assert row["source_tool"] == "hook:Read"
            assert row["chunk_id"] in chunk_ids

    def test_edit_tool_logs_interaction(self, db_path, tmp_dir):
        project_path = str(tmp_dir / "myproject")
        os.makedirs(project_path, exist_ok=True)
        file_rel = "src/utils.py"
        file_abs = os.path.join(project_path, file_rel)
        os.makedirs(os.path.dirname(file_abs), exist_ok=True)
        with open(file_abs, "w") as f:
            f.write("def add(a, b): return a + b")

        chunk_ids = _setup_project(db_path, project_path, file_rel)

        stdin_data = json.dumps({"tool_name": "Edit", "tool_input": {"file_path": file_abs}})
        result = _run_hook(stdin_data, db_path)
        assert result.returncode == 0

        interactions = _get_interactions(db_path)
        assert len(interactions) == 2
        for row in interactions:
            assert row["interaction"] == "edit"
            assert row["source_tool"] == "hook:Edit"
            assert row["chunk_id"] in chunk_ids

    def test_non_matching_tool_no_interaction(self, db_path, tmp_dir):
        project_path = str(tmp_dir / "myproject")
        os.makedirs(project_path, exist_ok=True)
        _setup_project(db_path, project_path, "src/app.py")

        stdin_data = json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}})
        result = _run_hook(stdin_data, db_path)
        assert result.returncode == 0

        interactions = _get_interactions(db_path)
        assert len(interactions) == 0

    def test_invalid_json_no_crash(self, db_path):
        result = _run_hook("not valid json {{{{", db_path)
        assert result.returncode == 0

    def test_file_not_in_any_project_no_interaction(self, db_path, tmp_dir):
        project_path = str(tmp_dir / "myproject")
        os.makedirs(project_path, exist_ok=True)
        _setup_project(db_path, project_path, "src/app.py")

        stdin_data = json.dumps(
            {
                "tool_name": "Read",
                "tool_input": {"file_path": "/some/other/path/file.py"},
            }
        )
        result = _run_hook(stdin_data, db_path)
        assert result.returncode == 0

        interactions = _get_interactions(db_path)
        assert len(interactions) == 0


# -- PreToolUse Hook --


def _run_pre_hook(stdin_data: str, db_path: str) -> subprocess.CompletedProcess:
    """Run the pre_tool_use hook as a subprocess."""
    env = os.environ.copy()
    env["DNOMIA_KNOWLEDGE_DB"] = db_path
    return subprocess.run(
        [sys.executable, "-m", "dnomia_knowledge.hooks.pre_tool_use"],
        input=stdin_data,
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )


def _create_file_with_lines(path: str, n: int) -> None:
    """Create a file with exactly n lines."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i in range(n):
            f.write(f"line {i}\n")


class TestPreToolUseHook:
    def test_read_small_file_allowed(self, db_path, tmp_dir):
        """File < 300 lines -> allow (exit 0, empty stdout)."""
        project_path = str(tmp_dir / "myproject")
        file_abs = os.path.join(project_path, "small.py")
        _create_file_with_lines(file_abs, 100)
        _setup_project(db_path, project_path, "small.py")

        stdin_data = json.dumps({"tool_name": "Read", "tool_input": {"file_path": file_abs}})
        result = _run_pre_hook(stdin_data, db_path)
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_read_large_file_in_indexed_project_blocked(self, db_path, tmp_dir):
        """File > 300 lines in registered project -> block with JSON."""
        project_path = str(tmp_dir / "myproject")
        file_abs = os.path.join(project_path, "big.py")
        _create_file_with_lines(file_abs, 500)
        _setup_project(db_path, project_path, "big.py")

        stdin_data = json.dumps({"tool_name": "Read", "tool_input": {"file_path": file_abs}})
        result = _run_pre_hook(stdin_data, db_path)
        assert result.returncode == 0
        response = json.loads(result.stdout)
        assert response["decision"] == "block"
        assert "500 lines" in response["reason"]
        assert "dnomia-knowledge" in response["reason"]

    def test_read_large_file_not_in_project_allowed(self, db_path, tmp_dir):
        """File > 300 lines NOT in any project -> allow."""
        project_path = str(tmp_dir / "myproject")
        os.makedirs(project_path, exist_ok=True)
        _setup_project(db_path, project_path, "app.py")

        # Create large file outside the project
        outside_file = str(tmp_dir / "outside" / "big.py")
        _create_file_with_lines(outside_file, 500)

        stdin_data = json.dumps({"tool_name": "Read", "tool_input": {"file_path": outside_file}})
        result = _run_pre_hook(stdin_data, db_path)
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_grep_in_indexed_project_blocked(self, db_path, tmp_dir):
        """Grep path matches project -> block with JSON."""
        project_path = str(tmp_dir / "myproject")
        os.makedirs(project_path, exist_ok=True)
        _setup_project(db_path, project_path, "app.py")

        stdin_data = json.dumps(
            {
                "tool_name": "Grep",
                "tool_input": {"path": project_path, "pattern": "def main"},
            }
        )
        result = _run_pre_hook(stdin_data, db_path)
        assert result.returncode == 0
        response = json.loads(result.stdout)
        assert response["decision"] == "block"
        assert "test-proj" in response["reason"]
        assert "def main" in response["reason"]

    def test_grep_outside_project_allowed(self, db_path, tmp_dir):
        """Grep path not in any project -> allow."""
        project_path = str(tmp_dir / "myproject")
        os.makedirs(project_path, exist_ok=True)
        _setup_project(db_path, project_path, "app.py")

        stdin_data = json.dumps(
            {
                "tool_name": "Grep",
                "tool_input": {"path": "/some/other/dir", "pattern": "foo"},
            }
        )
        result = _run_pre_hook(stdin_data, db_path)
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_non_matching_tool_allowed(self, db_path):
        """tool_name='Bash' -> no block, empty stdout."""
        stdin_data = json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}})
        result = _run_pre_hook(stdin_data, db_path)
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_invalid_json_no_crash(self, db_path):
        """Garbage input -> exit 0, no crash."""
        result = _run_pre_hook("not valid json {{{{", db_path)
        assert result.returncode == 0

    def test_no_db_allows_everything(self, tmp_dir):
        """Nonexistent DB path -> allows Read of large file."""
        project_path = str(tmp_dir / "myproject")
        file_abs = os.path.join(project_path, "big.py")
        _create_file_with_lines(file_abs, 500)

        nonexistent_db = str(tmp_dir / "nonexistent" / "nope.db")

        stdin_data = json.dumps({"tool_name": "Read", "tool_input": {"file_path": file_abs}})
        result = _run_pre_hook(stdin_data, nonexistent_db)
        assert result.returncode == 0
        assert result.stdout.strip() == ""
