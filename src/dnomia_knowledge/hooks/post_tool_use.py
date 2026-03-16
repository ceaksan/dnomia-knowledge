"""PostToolUse hook for Claude Code.

Reads tool invocation JSON from stdin and logs chunk interactions
for Read/Edit operations on files within registered projects.

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


def _find_project(file_path: str, projects: list[dict]) -> dict | None:
    """Find the registered project whose path is a parent of file_path."""
    best: dict | None = None
    best_len = 0
    for proj in projects:
        proj_path = proj["path"]
        # Normalize: ensure trailing slash for prefix check
        normalized = proj_path if proj_path.endswith("/") else proj_path + "/"
        if file_path.startswith(normalized) or file_path == proj_path:
            if len(proj_path) > best_len:
                best = proj
                best_len = len(proj_path)
    return best


def _resolve_file_path(file_path: str, project_path: str) -> str:
    """Convert absolute file_path to relative path from project root."""
    if file_path.startswith(project_path):
        rel = file_path[len(project_path) :]
        if rel.startswith("/"):
            rel = rel[1:]
        return rel
    return file_path


def main() -> None:
    try:
        raw = sys.stdin.read()
        data = json.loads(raw)

        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input", {})

        if tool_name not in ("Read", "Edit"):
            return

        file_path = tool_input.get("file_path", "")
        if not file_path or not os.path.isabs(file_path):
            return

        db_path = _get_db_path()
        if not os.path.exists(db_path):
            return

        from dnomia_knowledge.models import InteractionType
        from dnomia_knowledge.store import Store

        store = Store(db_path)
        try:
            projects = store.list_projects()
            project = _find_project(file_path, projects)
            if project is None:
                return

            project_id = project["id"]
            project_path = project["path"]

            rel_path = _resolve_file_path(file_path, project_path)
            chunk_ids = store.get_chunk_ids_for_file(project_id, rel_path)
            if not chunk_ids:
                return

            interaction = InteractionType.READ if tool_name == "Read" else InteractionType.EDIT
            batch = [
                (cid, interaction, f"hook:{tool_name}", project_id, rel_path) for cid in chunk_ids
            ]
            store.batch_log_interactions(batch)
        finally:
            store.close()

    except Exception:
        import traceback

        print(traceback.format_exc(), file=sys.stderr)


if __name__ == "__main__":
    main()
