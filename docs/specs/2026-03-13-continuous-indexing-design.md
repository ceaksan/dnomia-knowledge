# Continuous Indexing for dnomia-knowledge

**Date:** 2026-03-13
**Status:** Draft
**Replaces:** Watchdog daemon design (DEFER'd — 500MB persistent memory, overengineering, SQLite contention)

## Problem

dnomia-knowledge indexing is manual. The user must run `dnomia-knowledge index /path` after every change. This leads to stale indexes, missed search results, and wasted context tokens when Claude searches outdated data.

## Solution

Lightweight automatic indexing via git post-commit hooks + launchd periodic job. No daemon, no new dependencies, no persistent memory.

## Architecture

```
git commit → post-commit hook → dnomia-knowledge index . (background, mkdir lock)
launchd (5min) → dnomia-knowledge index-all --changed (fcntl lock)
```

Both paths use file locking to prevent concurrent execution.

## Components

### 1. `index-all` CLI command

Added to `cli.py`.

```bash
dnomia-knowledge index-all [--changed]
```

- Iterates all projects in SQLite `projects` table
- `--changed`: compares `git rev-parse HEAD` against stored `last_indexed_commit` per project. Skips projects where HEAD hasn't changed. For non-git projects (vault), falls back to checking if any file mtime is newer than `projects.last_indexed`.
- Acquires file lock before indexing. If lock held, exits silently (another index is running)
- After successful index, updates `projects.last_indexed` timestamp and `projects.last_indexed_commit` (for git projects)
- Runs, completes, exits. No persistent process.

### 2. `install-hooks` CLI command

Added to `cli.py`.

```bash
dnomia-knowledge install-hooks [--uninstall]
```

- Reads all project paths from `projects` table
- For each project with a `.git` directory:
  - If no `post-commit` hook exists: creates one
  - If hook exists: appends dnomia-knowledge block (delimited by markers for idempotency)
- `--uninstall`: removes the dnomia-knowledge block from all hooks
- Idempotent: safe to run multiple times

### 3. git post-commit hook

Installed by `install-hooks`. Template:

```bash
# --- dnomia-knowledge-start ---
DNOMIA_BIN="/Users/ceair/Desktop/DNM_Projects/dnomia-knowledge/.venv/bin/dnomia-knowledge"
if [ -x "$DNOMIA_BIN" ]; then
  (
    mkdir /tmp/dnomia-knowledge.lock 2>/dev/null || exit 0
    trap 'rmdir /tmp/dnomia-knowledge.lock' EXIT
    "$DNOMIA_BIN" index "$(git rev-parse --show-toplevel)" >/dev/null 2>&1
  ) &
fi
# --- dnomia-knowledge-end ---
```

Key properties:
- Full path to binary (git hooks have minimal PATH)
- `mkdir` lock: atomic on all POSIX systems, works on macOS (no `flock`)
- `trap` cleanup: removes lock dir on exit
- Quoted path (handles spaces in project paths)
- Background (`&`): doesn't block commit
- stdout/stderr redirected to `/dev/null` (no terminal noise after commit)
- Marker comments: `install-hooks --uninstall` can find and remove

Note: `install-hooks` writes the correct `DNOMIA_BIN` path dynamically based on the current installation.

### 4. `install-launchd` CLI command

Added to `cli.py`.

```bash
dnomia-knowledge install-launchd [--uninstall]
```

- Generates plist with correct `.venv/bin/dnomia-knowledge` path
- Copies to `~/Library/LaunchAgents/com.dnomia-knowledge.index.plist`
- Runs `launchctl load` (or `bootstrap` on newer macOS)
- `--uninstall`: unloads and removes plist
- Avoids hardcoded paths in repo files

Generated plist:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.dnomia-knowledge.index</string>
    <key>ProgramArguments</key>
    <array>
        <string>{dynamic_path_to_venv}/dnomia-knowledge</string>
        <string>index-all</string>
        <string>--changed</string>
    </array>
    <key>StartInterval</key>
    <integer>300</integer>
    <key>StandardOutPath</key>
    <string>{home}/.local/share/dnomia-knowledge/index.log</string>
    <key>StandardErrorPath</key>
    <string>{home}/.local/share/dnomia-knowledge/index-error.log</string>
</dict>
</plist>
```

No `RunAtLoad` (unnecessary, first run at next interval). No `KeepAlive` (not a daemon, just periodic).

### 5. File lock mechanism

Two compatible mechanisms for shell and Python:

**Shell (post-commit hook):**
- `mkdir /tmp/dnomia-knowledge.lock` — atomic, POSIX-portable, works on macOS
- Non-blocking: if dir exists, `mkdir` fails, hook exits silently
- Cleanup via `trap 'rmdir ...' EXIT`

**Python (CLI commands):**
- `fcntl.flock(fd, LOCK_EX | LOCK_NB)` on `/tmp/dnomia-knowledge.lockfile`
- Non-blocking: if lock held, exits with code 0 (not an error)
- Separate lock files for shell (dir) and Python (file) to avoid cross-mechanism issues

Guarantees: only one index process runs at any time, preventing OOM from concurrent 500MB model loads and SQLite write contention.

### 6. `projects` table updates

`store.py` needs two additions (no schema migration needed if columns already exist, otherwise ALTER TABLE):

- `update_project_last_indexed(project_id, timestamp)` — called at end of successful `index_directory()`
- `last_indexed_commit` column — stores `git rev-parse HEAD` for fast `--changed` detection

If `last_indexed_commit` column doesn't exist, add via `ALTER TABLE projects ADD COLUMN last_indexed_commit TEXT`.

## Files Changed

| File | Change |
|------|--------|
| `src/dnomia_knowledge/cli.py` | Add `index-all`, `install-hooks`, `install-launchd` commands |
| `src/dnomia_knowledge/indexer.py` | Add `index_all()` method, add file lock acquisition, update `last_indexed` after success |
| `src/dnomia_knowledge/store.py` | Add `update_project_last_indexed()`, add `last_indexed_commit` column migration |

## Files NOT Changed

- `server.py` — MCP tools unchanged
- `search.py` — search logic unchanged
- `pyproject.toml` — no new dependencies

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Commit during cron run | Hook tries mkdir lock, fails (dir exists), skips. Next cron catches it. |
| Cron during commit index | Same: fcntl lock prevents concurrent run. |
| Project deleted/moved | `index-all` catches FileNotFoundError, logs warning, skips project. |
| Large git checkout (many files) | Incremental indexer handles via MD5 hash — only changed files re-embedded. |
| dnomia-knowledge not installed | Hook checks `[ -x "$DNOMIA_BIN" ]`, exits silently. |
| New project added after install-hooks | Hook not present until `install-hooks` re-run. Cron still catches it via `index-all`. |
| Lock dir left behind (crash) | Stale mkdir lock blocks future hooks. Mitigation: cron uses separate fcntl lock, so periodic indexing continues. Stale dir cleaned on reboot (/tmp). |

## What This Does NOT Do

- Real-time indexing (sub-second latency) — not needed for current usage
- Cross-machine sync — single machine only
- Automatic project discovery — projects must be indexed once manually first
- Log rotation — use `newsyslog.d` or manual truncation
- Auto-install hooks on new project index — user runs `install-hooks` after adding projects

## Court Verdict

Evaluated by 3-juror tribunal (Claude, ChatGPT, Kimi). Original watchdog daemon design DEFER'd. This lightweight revision received GO with mandatory fixes incorporated:

1. `mkdir`-based lock for shell (macOS has no `flock` command)
2. launchd instead of cron (macOS Full Disk Access requirement)
3. Robust shell hook with full binary path, quoting, and output suppression
4. `last_indexed` / `last_indexed_commit` tracking for fast `--changed` detection
5. `install-launchd` CLI command (no hardcoded paths in repo)

## Spec Review

Reviewed by spec-document-reviewer. Critical fixes applied:
- Replaced `flock` with `mkdir`-based lock (macOS compatibility)
- Added `last_indexed` update after successful indexing
- Added `last_indexed_commit` for git-based change detection (faster than mtime scan)
- Replaced hardcoded plist with `install-launchd` CLI command
- Added full binary path in hook (git hooks have minimal PATH)
- Added stdout/stderr suppression in hook
