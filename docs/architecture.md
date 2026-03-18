# Architecture

## Overview

dnomia-knowledge is a unified knowledge management system. It indexes markdown and source code files into a single SQLite database, provides hybrid search (FTS5 keyword + vector semantic), and exposes results via an MCP server.

## Core Modules

### store.py

Single SQLite database backing everything. Schema versioned (current: v3), auto-migrates on startup.

Tables:

- `projects` — registered project directories
- `chunks` — indexed content chunks with embeddings
- `fts_chunks` — FTS5 virtual table mirroring chunks
- `file_index` — MD5 hash per file for incremental change detection
- `trace_reads`, `trace_edits`, `trace_searches` — usage interaction log
- `git_commits` — parsed git log entries (schema v3)
- `git_file_changes` — per-file diff stats per commit (schema v3)
- `git_sync_state` — sync cursor tracking last synced hash (schema v3)

### indexer.py

Orchestrates file indexing. Walks project directory, filters by extension, delegates chunking to `chunker/`, generates embeddings via `embedder.py`, writes to store. Incremental mode skips files where MD5 matches stored hash.

On each index run, also triggers an incremental git sync for the project (if it is a git repository).

### chunker/

- `markdown.py` — heading-based split (`##`/`###`), frontmatter parse, min 200 char merge
- `code.py` — Tree-sitter AST chunking per language

### embedder.py

Loads `intfloat/multilingual-e5-base` (768d). Prepends `query:` or `passage:` prefix per the model's convention. Lazy load, explicit unload to stay within 8GB RAM.

### search.py

Hybrid search: FTS5 BM25 + cosine KNN via sqlite-vec. Results merged with Reciprocal Rank Fusion (k=60). Prefix fallback for short queries.

### server.py

FastMCP server exposing three tools: `search`, `index_project`, `project_info`.

### hooks/

Python hooks for Claude Code tool lifecycle.

- `pre_tool_use.py` — intercepts Read/Grep calls, enriches context from knowledge base
- `post_tool_use.py` — intercepts Read/Edit calls, writes trace records

### graph.py

Knowledge graph utilities. Parses wikilinks and cross-references between chunks.

### registry.py

Manages the list of registered projects and their root paths.

### lock.py

File-based lock to prevent concurrent index runs (avoids OOM from multiple embedding model loads).

### presets.py

Named configuration presets for common project types (code, docs, obsidian vault).

---

## Git History Analysis

### git_sync.py

Parses git log output via `subprocess` and writes the results into the three git tables in SQLite.

**Sync strategy:**

1. On first sync (no stored state), parses full history with `git log --numstat`.
2. On subsequent syncs, uses `git merge-base` to find the divergence point and parses only new commits since the last synced hash.
3. `--force` flag bypasses merge-base and triggers a full resync from scratch.

**`GitSync` class responsibilities:**

- Shell out to `git log --format` and `--numstat` to collect commit metadata and per-file diff stats.
- Detect renames via `--diff-filter=R` and store both old and new file paths.
- Write parsed records to `git_commits` and `git_file_changes`.
- Update `git_sync_state` with the latest synced hash after each successful run.

**Data flow:**

```
git log (subprocess) -> parse output -> GitSync -> store.py (git CRUD) -> SQLite
```

### git_analyze.py

Crossover signal classification. Combines trace read data with git churn data to produce a signal per file.

**Thresholds:** P75 computed from the query result set itself, not global constants. This makes the classification relative to the current project and time window.

**Signal definitions:**

| Signal | Condition |
|--------|-----------|
| BLIND | High churn, zero reads |
| TURBULENT | High churn, low reads (non-zero) |
| HOT | High churn, high reads |
| STABLE | Low churn, high reads |
| ZOMBIE | Zero churn, some reads |
| COLD | Low churn, low reads |

`classify_crossover_results(rows)` accepts a list of dicts with `churn` and `reads` keys, computes thresholds, and returns the same list with a `signal` field added to each row.

---

## Schema v3 Tables

```sql
CREATE TABLE git_commits (
    project_id TEXT NOT NULL,
    hash TEXT NOT NULL,
    timestamp INTEGER NOT NULL,   -- Unix epoch
    summary TEXT,
    PRIMARY KEY (project_id, hash)
);

CREATE TABLE git_file_changes (
    project_id TEXT NOT NULL,
    commit_hash TEXT NOT NULL,
    file_path TEXT NOT NULL,
    old_file_path TEXT,           -- populated on renames
    insertions INTEGER DEFAULT 0,
    deletions INTEGER DEFAULT 0,
    change_type TEXT,             -- A, M, D, R, etc.
    is_binary INTEGER DEFAULT 0,
    PRIMARY KEY (project_id, commit_hash, file_path)
);

CREATE TABLE git_sync_state (
    project_id TEXT PRIMARY KEY,
    last_synced_hash TEXT,
    last_synced_at INTEGER,       -- Unix epoch
    total_commits INTEGER DEFAULT 0
);
```

---

## Data Flows

### Index flow

```
CLI / MCP -> indexer.index_directory()
  -> walk files -> chunker -> embedder -> store (chunks, fts, file_index)
  -> git_sync.sync() [incremental]
       -> git log (subprocess) -> store (git_commits, git_file_changes, git_sync_state)
```

### Search flow

```
MCP search tool -> HybridSearch.search()
  -> FTS5 query (BM25 scores) + vec_search (cosine KNN)
  -> RRF merge -> return ranked chunks
  -> trace_searches record written
```

### Analyze flow

```
CLI analyze crossover -> store.query_crossover()
  -> JOIN git_file_changes + trace_reads
  -> git_analyze.classify_crossover_results()
       -> compute P75 thresholds -> assign signal per file
  -> tabulate and print
```

---

## Continuous Indexing Integration

The `indexer.py` incremental sync hook means git history stays current automatically:

- Every `index_project` MCP call triggers `GitSync.sync()` for the project.
- The git post-commit hook (installed via `install-hooks`) calls `index_project` on commit, so git history is synced within seconds of a new commit.
- The launchd periodic job (every 5 minutes) also calls `index-all`, which triggers git sync on all registered projects.

No separate cron or daemon needed for git history.
