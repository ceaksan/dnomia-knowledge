"""Tests for PostToolUse hook script."""

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
