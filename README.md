# dnomia-knowledge

Unified knowledge management MCP server. Indexes markdown and code files with hybrid search (FTS5 keyword + vector semantic).

SQLite + sqlite-vec + FTS5 single DB. multilingual-e5-base embedding. Heading-based markdown chunking + Tree-sitter AST code chunking.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `intfloat/multilingual-e5-base` model (~500MB) is automatically downloaded on first run.

## MCP Configuration

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "dnomia-knowledge": {
      "command": "/path/to/dnomia-knowledge/.venv/bin/python",
      "args": ["-m", "dnomia_knowledge.server"],
      "env": {
        "DNOMIA_KNOWLEDGE_PROJECT": "my-project"
      }
    }
  }
}
```

Can also be added to a project's `.claude/settings.json` for project-level configuration.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DNOMIA_KNOWLEDGE_DB` | `~/.local/share/dnomia-knowledge/knowledge.db` | SQLite DB file path |
| `DNOMIA_KNOWLEDGE_PROJECT` | (none) | Default project ID for search tool |

## MCP Tools

### search

Hybrid semantic + keyword search.

```
search(query, domain="all", project=None, limit=10)
```

- `query`: Search text
- `domain`: `"all"`, `"content"`, or `"code"`
- `project`: Project ID. Falls back to `DNOMIA_KNOWLEDGE_PROJECT` env var if not specified
- `limit`: Maximum number of results

### index_project

Index a project directory.

```
index_project(path, incremental=True)
```

- `path`: Project root directory (absolute path)
- `incremental`: When `True`, only re-indexes changed files

Project ID is derived from directory name (lowercase, spaces become hyphens).

### project_info

List registered projects and their statistics.

```
project_info(project=None)
```

- `project`: Specific project ID, or `None` for all projects

## Usage (Python)

```python
from dnomia_knowledge.store import Store
from dnomia_knowledge.embedder import Embedder
from dnomia_knowledge.indexer import Indexer
from dnomia_knowledge.search import HybridSearch

store = Store("/tmp/test.db")
embedder = Embedder()
indexer = Indexer(store, embedder)

# Index
result = indexer.index_directory("my-project", "/path/to/project")
print(f"{result.total_chunks} chunks, {result.total_files} files")

# Search
search = HybridSearch(store, embedder)
results = search.search("JWT authentication", project_id="my-project")
for r in results:
    print(f"[{r.score:.4f}] {r.file_path} - {r.name}")

store.close()
```

## Continuous Indexing

Automatic index updates via git post-commit hooks and launchd periodic job. No daemon, no extra dependencies, no persistent memory usage.

### Setup

```bash
# Install git hooks on all indexed projects
dnomia-knowledge install-hooks

# Install launchd periodic job (every 5 min)
dnomia-knowledge install-launchd
```

### How it works

- **Git hooks**: After every commit, the changed project is re-indexed in the background
- **launchd job**: Every 5 minutes, checks all registered projects for changes and re-indexes if needed
- **File lock**: Only one index process runs at a time (prevents OOM from concurrent embedding model loads)
- **Change detection**: Git HEAD hash comparison for git projects, recursive mtime scan for non-git directories (Obsidian vault)

### CLI Commands

```bash
# Index all registered projects
dnomia-knowledge index-all

# Only index projects with changes since last index
dnomia-knowledge index-all --changed

# Install/remove git post-commit hooks
dnomia-knowledge install-hooks
dnomia-knowledge install-hooks --uninstall

# Install/remove launchd periodic job
dnomia-knowledge install-launchd
dnomia-knowledge install-launchd --uninstall
```

### Verify

```bash
# Check launchd is running
launchctl list | grep dnomia-knowledge

# Check logs
tail -f ~/.local/share/dnomia-knowledge/index.log
```

## Git History Analysis

### git-sync

Parses git log and stores commit history and per-file diff stats in SQLite.

```bash
# Sync git history for a project (incremental after first run)
dnomia-knowledge git-sync /path/to/project

# Force full resync (re-parses entire git history)
dnomia-knowledge git-sync /path/to/project --force
```

First run parses full history. Subsequent runs are incremental: only new commits since the last synced hash are parsed. `--force` bypasses this and re-parses from scratch.

**Prerequisites:** The target path must be a git repository. Git must be available in PATH.

### analyze

Analyze git history data. All subcommands share the same options:

| Flag | Default | Description |
|------|---------|-------------|
| `--project` / `-p` | all | Filter by project ID |
| `--days` / `-d` | 90 | Time window in days |
| `--limit` / `-l` | 20 | Max rows to display |

#### churn

Most modified files by total churn (insertions + deletions).

```bash
dnomia-knowledge analyze churn
dnomia-knowledge analyze churn -p my-project -d 30 -l 10
```

#### hotspots

Directory-level churn aggregation. Groups file churn by parent directory to surface which areas of the codebase change most.

```bash
dnomia-knowledge analyze hotspots
dnomia-knowledge analyze hotspots -p my-project -d 60
```

#### crossover

Fuses git churn data with trace read data. Assigns a signal to each file based on how much it changes versus how often it is read.

```bash
dnomia-knowledge analyze crossover
dnomia-knowledge analyze crossover -p my-project -d 90 -l 30
```

Signals use P75-based thresholds computed from the result set:

| Signal | Meaning |
|--------|---------|
| BLIND | High churn, zero reads. Actively changing but never consulted. |
| TURBULENT | High churn, low reads. Unstable and under-monitored. |
| HOT | High churn, high reads. Core active area. |
| STABLE | Low churn, high reads. Settled reference code. |
| ZOMBIE | Zero churn, some reads. Read but never touched. |
| COLD | Low churn, low reads. Inactive. |

## Trace Analytics

Usage pattern analysis from accumulated interaction and search data. Requires no extra setup: data is collected automatically as you use search and read/edit files.

```bash
# Most accessed files
dnomia-knowledge trace hot

# Searches that returned no results (knowledge gaps)
dnomia-knowledge trace gaps

# Files with declining activity
dnomia-knowledge trace decay

# Most frequent search patterns
dnomia-knowledge trace queries
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--project` / `-p` | all | Filter by project ID |
| `--days` / `-d` | 30 | Time window in days |
| `--limit` / `-l` | 20 | Max rows to display |

```bash
# Last 7 days, specific project, top 5
dnomia-knowledge trace hot -p my-project -d 7 -l 5
```

## Claude Code Hooks

### Pre/Post Tool Use Hooks

Python-based hooks that integrate with Claude Code's tool lifecycle:

- **PreToolUse (Read|Grep)**: Enriches file reads with knowledge base context
- **PostToolUse (Read|Edit)**: Tracks file interactions for trace analytics

Configured in `~/.claude/settings.json` under `hooks.PreToolUse` and `hooks.PostToolUse`.

### WIP Persistence Hooks

Shell scripts that prevent work-in-progress loss during context compaction:

- **PreCompact (`scripts/checkpoint-wip.sh`)**: Saves current branch, modified files, git status, and active tasks to `~/.claude/state/` before context is compressed
- **SessionStart (`scripts/restore-wip.sh`)**: Restores checkpoint as `additionalContext` when a new session starts, so Claude knows where you left off

Checkpoints auto-expire after 24 hours and are removed after one restore.

```bash
# Checkpoint format (~/.claude/state/wip-{project}.json)
{
  "project": "content-intelligence",
  "branch": "main",
  "modified_files": ["src/actions/link_validator.py"],
  "git_status": "1 file changed, 95 insertions(+)",
  "timestamp": "2026-03-17T08:12:42Z"
}
```

## Technical Details

- **DB**: SQLite + FTS5 (BM25 keyword) + sqlite-vec (cosine KNN)
- **Embedding**: intfloat/multilingual-e5-base (768d), `query:`/`passage:` prefix required
- **Search**: Hybrid FTS5 + vector, RRF merge (k=60), prefix fallback
- **Chunking**: `##`/`###` heading split, frontmatter parsing, min 200 char merge, overlap support
- **Incremental**: MD5 content hash for change detection
- **Memory**: Lazy model loading, explicit unload, runs on 8GB RAM
- **Schema**: v3, auto-migrates from v1/v2 on first run. v3 adds git_commits, git_file_changes, git_sync_state tables.

## Testing

```bash
source .venv/bin/activate
python -m pytest tests/ -v
ruff check src/ tests/
```

## License

MIT
