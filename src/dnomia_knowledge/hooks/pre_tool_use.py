"""PreToolUse hook for context minimization.

Intercepts Read/Grep tool calls and blocks them when an indexed
alternative exists, guiding Claude to use MCP search tools instead.

Must never crash or return non-zero exit code.
"""

from __future__ import annotations

import json
import os
import sys


def _get_db_path() -> str:
    return os.environ.get(
        "DNOMIA_KNOWLEDGE_DB",
        os.path.expanduser("~/.local/share/dnomia-knowledge/knowledge.db"),
    )


def _count_file_lines(file_path: str) -> int:
    try:
        with open(file_path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _find_project_for_path(file_path: str, projects: list[dict]) -> dict | None:
    """Find registered project containing this file (longest path match)."""
    best = None
    best_len = 0
    for proj in projects:
        proj_path = proj["path"]
        normalized = proj_path if proj_path.endswith("/") else proj_path + "/"
        if file_path.startswith(normalized) or file_path == proj_path:
            if len(proj_path) > best_len:
                best = proj
                best_len = len(proj_path)
    return best


def _deny(reason: str) -> None:
    response = {"decision": "block", "reason": reason}
    print(json.dumps(response))
    sys.exit(0)


def _handle_read(tool_input: dict) -> None:
    file_path = tool_input.get("file_path", "")
    if not file_path or not os.path.isfile(file_path):
        return

    line_count = _count_file_lines(file_path)
    if line_count <= 300:
        return

    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return

    from dnomia_knowledge.store import Store

    store = Store(db_path)
    try:
        projects = store.list_projects()
        project = _find_project_for_path(file_path, projects)
        if project is None:
            return

        _deny(
            f"File has {line_count} lines. "
            f'Use dnomia-knowledge read_file tool with file_path="{file_path}" '
            f"and a query parameter, or use dnomia-knowledge search tool."
        )
    finally:
        store.close()


def _handle_grep(tool_input: dict) -> None:
    path = tool_input.get("path", "")
    if not path:
        return

    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return

    from dnomia_knowledge.store import Store

    store = Store(db_path)
    try:
        projects = store.list_projects()
        project = _find_project_for_path(path, projects)
        if project is None:
            # Check if path is a parent of any project
            for proj in projects:
                if proj["path"].startswith(path.rstrip("/") + "/") or proj["path"] == path:
                    project = proj
                    break

        if project is None:
            return

        pattern = tool_input.get("pattern", "")
        _deny(
            f"Project '{project['id']}' is indexed. "
            f"Use dnomia-knowledge search tool with query: '{pattern}'"
        )
    finally:
        store.close()


def main() -> None:
    try:
        raw = sys.stdin.read()
        data = json.loads(raw)
        tool_name = data.get("tool_name", "")

        if tool_name == "Read":
            _handle_read(data.get("tool_input", {}))
        elif tool_name == "Grep":
            _handle_grep(data.get("tool_input", {}))
    except Exception:
        pass


if __name__ == "__main__":
    main()
