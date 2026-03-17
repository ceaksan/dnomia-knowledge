"""Git log parsing and sync engine."""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass

_COMMIT_HASH_RE = re.compile(r"[0-9a-f]{40}")
_BRACE_RENAME_RE = re.compile(r"^(.*)\{(.*) => (.*)\}(.*)$")
_SIMPLE_RENAME_RE = re.compile(r"^(.*) => (.*)$")
_REF_RE = re.compile(r"^[0-9a-f]{4,40}$")


@dataclass
class SyncResult:
    project_id: str
    mode: str
    commits_parsed: int
    changes_parsed: int
    duration_seconds: float


def _safe_env() -> dict:
    """Build env dict for locale-independent, non-interactive git."""
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["GIT_PAGER"] = "cat"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _run_git(args: list[str], cwd: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Safe subprocess wrapper for git commands."""
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_safe_env(),
        start_new_session=True,
    )


def validate_ref(ref: str) -> bool:
    """Validate SHA format: 4-40 hex chars, no dashes."""
    if not ref or ref.startswith("-"):
        return False
    return bool(_REF_RE.match(ref))


def is_git_repo(path: str) -> bool:
    """Check if path is inside a git repository."""
    try:
        result = _run_git(["git", "rev-parse", "--git-dir"], cwd=path, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def get_head_hash(repo_path: str) -> str | None:
    """Get HEAD SHA."""
    try:
        result = _run_git(["git", "rev-parse", "HEAD"], cwd=repo_path, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, OSError):
        return None


def get_commit_count(repo_path: str) -> int:
    """Get total commit count."""
    try:
        result = _run_git(
            ["git", "rev-list", "--count", "HEAD", "--end-of-options"],
            cwd=repo_path,
            timeout=30,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
        return 0
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return 0


def parse_numstat_line(line: str) -> dict | None:
    """Parse a single numstat line.

    Returns dict with file_path, old_file_path, insertions, deletions,
    change_type (M/R), is_binary. Returns None for empty/unparseable lines.
    """
    if not line or not line.strip():
        return None

    parts = line.split("\t", 2)
    if len(parts) != 3:
        return None

    ins_str, del_str, file_part = parts

    # Binary detection
    is_binary = ins_str == "-" and del_str == "-"
    insertions = None if is_binary else int(ins_str)
    deletions = None if is_binary else int(del_str)

    # Rename detection
    file_path = file_part
    old_file_path = None
    change_type = "M"

    # Brace rename: src/{old => new}/file.py
    brace_match = _BRACE_RENAME_RE.match(file_part)
    if brace_match:
        prefix, old_mid, new_mid, suffix = brace_match.groups()
        old_file_path = (prefix + old_mid + suffix).strip()
        file_path = (prefix + new_mid + suffix).strip()
        change_type = "R"
    else:
        # Simple rename: old.py => new.py
        simple_match = _SIMPLE_RENAME_RE.match(file_part)
        if simple_match:
            old_file_path = simple_match.group(1).strip()
            file_path = simple_match.group(2).strip()
            change_type = "R"

    return {
        "file_path": file_path,
        "old_file_path": old_file_path,
        "insertions": insertions,
        "deletions": deletions,
        "change_type": change_type,
        "is_binary": 1 if is_binary else 0,
    }


def parse_git_log_output(raw: str, project_id: str) -> tuple[list[dict], list[dict]]:
    """Parse raw git log output into commits and file changes.

    Splits by commit headers (40-char hex hash followed by \\x00).
    Returns (commits, file_changes).
    """
    if not raw or not raw.strip():
        return [], []

    commits: list[dict] = []
    changes: list[dict] = []

    # Split on commit boundaries: 40-char hex hash followed by \x00
    # Use lookahead to keep the hash in the split result
    sections = re.split(r"(?=^[0-9a-f]{40}\x00)", raw, flags=re.MULTILINE)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # First line should be: hash\x00timestamp\x00summary
        lines = section.split("\n")
        header = lines[0]

        header_parts = header.split("\x00", 2)
        if len(header_parts) < 3:
            continue

        commit_hash, timestamp_str, summary = header_parts

        if not _COMMIT_HASH_RE.fullmatch(commit_hash):
            continue

        try:
            timestamp = int(timestamp_str)
        except ValueError:
            continue

        commits.append(
            {
                "project_id": project_id,
                "hash": commit_hash,
                "timestamp": timestamp,
                "summary": summary[:500],
            }
        )

        # Parse numstat lines
        for numstat_line in lines[1:]:
            parsed = parse_numstat_line(numstat_line)
            if parsed:
                parsed["project_id"] = project_id
                parsed["commit_hash"] = commit_hash
                changes.append(parsed)

    return commits, changes


def detect_sync_strategy(repo_path: str, last_synced_hash: str) -> str:
    """Check if last_synced_hash is ancestor of HEAD.

    Returns 'incremental' or 'full'.
    """
    if not validate_ref(last_synced_hash):
        return "full"

    try:
        result = _run_git(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                last_synced_hash,
                "HEAD",
                "--end-of-options",
            ],
            cwd=repo_path,
            timeout=30,
        )
        return "incremental" if result.returncode == 0 else "full"
    except (subprocess.TimeoutExpired, OSError):
        return "full"


def get_git_log(repo_path: str, since_hash: str | None = None) -> str:
    """Run git log with numstat output.

    Returns raw output string.
    """
    args = [
        "git",
        "log",
        "--numstat",
        "--format=%H\x00%at\x00%s",
        "--no-merges",
        "-M",
    ]

    if since_hash and validate_ref(since_hash):
        args.append(f"{since_hash}..HEAD")

    args.append("--end-of-options")

    result = _run_git(args, cwd=repo_path)
    if result.returncode != 0:
        return ""
    return result.stdout


class GitSync:
    """Sync engine for git history data."""

    def __init__(self, store):
        self.store = store

    def sync(self, project_id: str, repo_path: str, force: bool = False) -> SyncResult:
        """Full or incremental sync of git history."""
        start = time.monotonic()

        state = self.store.get_git_sync_state(project_id)
        last_hash = state.get("last_synced_hash") if state else None

        if force or not last_hash:
            mode = "full"
        else:
            mode = detect_sync_strategy(repo_path, last_hash)

        if mode == "full":
            self.store.clear_git_data(project_id)
            raw = get_git_log(repo_path)
        else:
            raw = get_git_log(repo_path, since_hash=last_hash)

        commits, changes = parse_git_log_output(raw, project_id)

        if commits:
            self.store.save_git_commits(commits)
        if changes:
            self.store.save_git_file_changes(changes)

        head = get_head_hash(repo_path)
        total = get_commit_count(repo_path)
        if head:
            self.store.update_git_sync_state(project_id, head, total)

        duration = time.monotonic() - start
        return SyncResult(
            project_id=project_id,
            mode=mode,
            commits_parsed=len(commits),
            changes_parsed=len(changes),
            duration_seconds=round(duration, 2),
        )

    def sync_incremental(self, project_id: str, repo_path: str) -> SyncResult | None:
        """Incremental sync for use during index_directory().

        Returns None if no git repo, not synced yet, or already up to date.
        """
        if not is_git_repo(repo_path):
            return None

        state = self.store.get_git_sync_state(project_id)
        if not state or not state.get("last_synced_hash"):
            return None

        head = get_head_hash(repo_path)
        if not head or head == state.get("last_synced_hash"):
            return None

        return self.sync(project_id, repo_path, force=False)
