# dnomia-knowledge Architecture

Local knowledge engine for codebases. Hybrid search (FTS5 + sqlite-vec), knowledge graph (Louvain + PageRank), and developer interaction tracking over a single SQLite database.

Last verified: 2026-03-18

## Stack

| Package | Version | Purpose |
|---------|---------|---------|
| mcp[cli] | >=1.13.0 | FastMCP server framework (stdio transport) |
| sentence-transformers | >=3.0.0 | intfloat/multilingual-e5-base embeddings (768d) |
| sqlite-vec | >=0.1.6 | Vector KNN search extension for SQLite |
| tree-sitter | >=0.24.0 | AST parsing for code chunking |
| tree-sitter-language-pack | >=0.7.0 | Language grammars (Python, JS/TS, Go, Rust, etc.) |
| networkx | >=3.0 | Knowledge graph (Louvain communities, PageRank) |
| pydantic | >=2.0.0 | Config validation (.knowledge.toml schema) |
| python-frontmatter | >=1.0.0 | YAML frontmatter extraction from markdown |
| pathspec | >=0.12.0 | .gitignore pattern matching |
| rich | >=13.0.0 | CLI tables and progress bars |

### Infrastructure

| Layer | Technology | Detail |
|-------|-----------|--------|
| Database | SQLite + WAL | Single file, 64MB mmap, 64K cache |
| Full-text search | FTS5 | BM25 ranking, porter stemmer, unicode61 tokenizer |
| Vector search | sqlite-vec | Cosine KNN on 768d normalized vectors |
| Embedding model | multilingual-e5-base | ~500MB, lazy loaded, auto-unloads after 10min idle |
| MCP transport | stdio | FastMCP server over stdin/stdout |
| Continuous indexing | git hooks + launchd | Post-commit hook + 5-min periodic job, no daemon |

## Module Map

```
src/dnomia_knowledge/
  server.py          MCP server, 6 tools, thread-safe singletons (RLock)
  store.py           SQLite persistence, schema v3, CRUD, triggers, migrations
                     Public API: execute_sql(), fetchall(), commit()
  embedder.py        Lazy sentence-transformer, query/passage prefix, LRU cache
  indexer.py         Scan -> filter -> chunk -> embed -> store pipeline
  search.py          Hybrid FTS5+vector, RRF merge, interaction boost, cross-project
  graph.py           Edge builder (link/tag/category/semantic/import), Louvain, PageRank
  lock.py            fcntl file lock for preventing concurrent index runs
  cli.py             Rich CLI: index, search, graph, trace, git-sync, analyze, hooks, launchd
  registry.py        .knowledge.toml config loader, Pydantic v2 schema
  presets.py         Extension presets (web, python, django, mixed)
  models.py          Chunk, SearchResult, IndexResult, InteractionType, ChunkType
  git_sync.py        Git log parser, incremental commit history sync
  git_analyze.py     Churn, hotspot, crossover analysis over git history
  chunker/
    base.py          Chunker protocol definition
    md_chunker.py    Heading-based markdown splitter (##/###, frontmatter, min 200 char)
    ast_chunker.py   Tree-sitter AST chunker with sliding-window fallback
    languages.py     Per-language AST node type configs
  hooks/
    pre_tool_use.py  Blocks Read/Grep on indexed large files (>300 lines), redirects to search
    post_tool_use.py Logs read/edit interactions for search ranking boost
```

```
tests/ (276 tests)
  conftest.py           Shared fixtures (tmp_dir, tmp_store, embedder, indexer)
  test_store.py         SQLite ops, triggers, indexes, wrapper methods
  test_embedder.py      Lazy loading, caching, batching
  test_chunker.py       Heading-based markdown splitting
  test_ast_chunker.py   AST extraction, language support
  test_indexer.py       File scanning, incremental detection
  test_search.py        FTS5, vector, RRF merge, interaction boost
  test_graph.py         Edge building, Louvain, PageRank, BFS
  test_server.py        MCP tool interface, symlink protection
  test_cli.py           CLI command handlers
  test_hooks.py         Pre/post tool use hooks
  test_integration.py   End-to-end flows
  test_continuous.py    Git hooks, launchd, change detection
  test_git_sync.py      Git log parsing, incremental sync
  test_lock.py          File locking
  test_registry.py      Config loading
  test_presets.py        Extension preset resolution
```

## Data Flow

### Indexing

```
Project directory
  -> Scan: filter binary/size/ignore (.gitignore + config patterns)
  -> For each changed file (MD5 hash comparison):
      -> Route by extension: .md/.mdx -> MdChunker | code -> AstChunker
      -> Chunk list (name, type, lines, content, metadata)
      -> Embed passages (batch=8, "passage: " prefix)
      -> Atomic transaction: delete old -> insert chunks -> insert vectors -> update file_index
  -> Build graph edges (if enabled)
  -> Incremental git sync (if previously synced)
  -> Update projects.last_indexed + last_indexed_commit
```

### Search

```
Query text
  -> embed_query() (cached, "query: " prefix, 768d vector)
  -> Parallel:
      FTS5 BM25 search -> ranked list A
      sqlite-vec KNN   -> ranked list B
  -> RRF merge (k=60): score = sum(1/(k + rank + 1))
  -> If empty: retry FTS with prefix matching (word*)
  -> Interaction boost: re-rank by read/edit frequency (30-day window)
  -> Return top N with snippets
```

### Trace Analytics

```
chunk_interactions + search_log (accumulated over time)
  -> trace hot:     GROUP BY (project_id, file_path), count R/E/S interactions
  -> trace gaps:    search_log WHERE result_count = 0, GROUP BY query
  -> trace decay:   compare interaction counts between two time windows
  -> trace queries: GROUP BY query, AVG result_count
```

### Git History Analysis

```
git log --numstat
  -> Parse commits (hash, author, date, message)
  -> Parse file changes (insertions, deletions per file)
  -> Store in git_commits + git_file_changes tables
  -> Incremental: only new commits since last sync

analyze churn:     SUM(insertions + deletions) per file
analyze hotspots:  Churn aggregated by directory
analyze crossover: Fuse git churn with trace read counts, classify signals
```

### Continuous Indexing

```
git commit -> post-commit hook -> mkdir lock -> dnomia-knowledge index . (background)
launchd (5min) -> index-all --changed -> fcntl lock -> git HEAD comparison -> re-index changed
```

## MCP Tools

```
search(query, domain, project, cross, limit, language, file_pattern, show_content)
  Hybrid FTS5+vector search, returns ranked results with snippets

index_project(path, incremental)
  Index directory, auto-detects .knowledge.toml config

project_info(project)
  List projects with chunk/file counts, graph status, last indexed

graph_query(project, mode, chunk_id, depth)
  mode=neighbors: BFS traversal | mode=communities: Louvain clusters

read_file(path, query, project)
  Smart file reader: large files return relevant chunks or metadata

fetch_and_index(url, project)
  HTTP fetch -> strip HTML -> chunk -> embed -> searchable immediately
```

## Data Model

### Core Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `projects` | Registered project metadata | id, path, type, graph_enabled, last_indexed, last_indexed_commit, config_hash |
| `chunks` | Indexed content/code pieces | id, project_id, file_path, chunk_domain, chunk_type, name, language, start_line, end_line, content, metadata (JSON) |
| `chunks_vec` | Vector embeddings (sqlite-vec virtual) | id, embedding (768d float32) |
| `chunks_fts` | Full-text index (FTS5 virtual) | content mirror, BM25 ranking |
| `file_index` | Per-file tracking for incremental indexing | project_id, file_path, file_hash (MD5), chunk_count, last_indexed |

### Graph

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `edges` | Knowledge graph relationships | source_id, target_id, edge_type (link/tag/category/semantic/import), weight, metadata |

### Analytics

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `chunk_interactions` | Usage tracking (InteractionType enum: read/edit/search_hit) | chunk_id, project_id, file_path, interaction, source_tool, timestamp |
| `search_log` | Query analytics | query, project_id, domain, result_chunk_ids, result_count, timestamp |

### Git History

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `git_commits` | Parsed git log entries | project_id, hash, author, date, message |
| `git_file_changes` | Per-file diff stats | commit_hash, file_path, insertions, deletions |
| `git_sync_state` | Incremental sync tracking | project_id, last_synced_hash |

### System

| Table | Purpose |
|-------|---------|
| `system_metadata` | embedding_model, embedding_dim, schema_version |

### Triggers

- `chunks INSERT` -> auto-insert to `chunks_fts`
- `chunks UPDATE` -> auto-update `chunks_fts`
- `chunks DELETE` -> auto-delete from `chunks_fts` + `chunks_vec`

### Indexes

- `chunks(project_id, file_path)` for file-level lookups
- `chunks(project_id, chunk_domain)` for domain filtering
- `file_index(project_id, file_path)` UNIQUE for incremental detection
- `edges(source_id)`, `edges(target_id)` for graph traversal
- `chunk_interactions(project_id, timestamp)` for trace queries
- `chunk_interactions(project_id, file_path)` for file-level aggregation
- `search_log(result_count, timestamp)` for knowledge gaps
- `search_log(query)` for query pattern analysis

## Internal Design Decisions

### Store Encapsulation

External modules access the database through `Store.execute_sql()`, `Store.fetchall()`, and `Store.commit()` instead of touching `Store._connect()` directly. This keeps the connection management internal and makes refactoring safe.

### Thread Safety

Server singletons (`_get_store`, `_get_embedder`, `_get_search`, `_get_indexer`) use `threading.RLock()` with double-checked locking. RLock (not Lock) because `_get_search()` calls `_get_store()` and `_get_embedder()` inside the lock.

### Exception Handling

Core modules (search, indexer, graph) use specific exception types (`sqlite3.Error`, `OSError`, `ValueError`) instead of bare `except Exception`. Hooks are the exception: they catch everything because a hook crash must never kill a Claude Code session. Hook errors are logged to stderr.

### InteractionType Enum

`InteractionType(StrEnum)` with values `READ`, `EDIT`, `SEARCH_HIT`. Used in search.py and hooks instead of magic strings. SQL queries still use raw string literals (SQLite needs them).

### Vector Validation

`insert_chunk_vectors()` raises `ValueError` if `len(chunk_ids) != len(vectors)`. Prevents silent zip truncation that would leave chunks without embeddings.

## Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `DNOMIA_KNOWLEDGE_DB` | SQLite database file path | `~/.local/share/dnomia-knowledge/knowledge.db` |
| `DNOMIA_KNOWLEDGE_PROJECT` | Default project ID for MCP search tool | (none) |

### Project Config (.knowledge.toml)

```toml
[project]
name = "my-project"
type = "saas"              # content | saas | static

[content]
paths = ["docs/"]
extensions = [".md", ".mdx"]
chunking = "heading"       # heading | sliding

[code]
preset = "python"          # web | python | django | mixed
paths = ["src/"]
max_chunk_lines = 50

[graph]
enabled = true
edge_types = ["link", "tag", "semantic", "import"]
semantic_threshold = 0.75

[indexing]
ignore_patterns = ["*.log", "migrations/"]
max_file_size_kb = 500
```

## Security

- File access restricted to registered project paths (read_file validates resolved symlinks)
- No network access except explicit fetch_and_index (user-initiated URL fetch)
- SQLite WAL mode prevents corruption from concurrent reads
- fcntl/mkdir file locks prevent concurrent writes (OOM protection)
- No authentication layer (local-only MCP server, stdio transport)
- Hooks never crash Claude Code session (catch-all with stderr logging)

## Dependency Graph

```
server.py -> store.py, embedder.py, indexer.py, search.py
indexer.py -> store.py, embedder.py, chunker/, graph.py, registry.py, lock.py, git_sync.py
search.py -> store.py, embedder.py
graph.py -> store.py
git_sync.py -> store.py
git_analyze.py -> store.py
cli.py -> store.py, embedder.py, indexer.py, search.py, lock.py, git_sync.py, git_analyze.py
registry.py -> presets.py, models.py
chunker/ast_chunker.py -> chunker/languages.py, chunker/base.py
chunker/md_chunker.py -> chunker/base.py
hooks/ -> store.py, models.py (lightweight, no embedder dependency)
```

No circular dependencies. `store.py` is the leaf dependency. `embedder.py` is independent.

## Constraints and Trade-offs

| Decision | Reason | Trade-off |
|----------|--------|-----------|
| SQLite single file | Zero ops, no daemon, portable | No concurrent multi-process writes, single machine only |
| multilingual-e5-base (768d) | Good multilingual support, reasonable size | ~500MB memory when loaded, not best-in-class for English-only |
| Lazy model loading | Avoid 500MB cost on every CLI/server start | First search has cold-start latency (~5s) |
| FTS5 + vector hybrid (RRF) | Better recall than either alone | Two-pass search adds latency vs single-pass |
| Tree-sitter AST chunking | Semantic code boundaries, language-aware | Adds ~100MB dependency, falls back to sliding-window on unsupported |
| git hooks + launchd (not watchdog) | No persistent process, no memory overhead | Max 5-min staleness between commits, not sub-second |
| mkdir lock (shell) + fcntl (Python) | macOS has no flock command, mkdir is atomic POSIX | Two separate lock mechanisms, stale mkdir lock blocks until reboot |
| Heading-based markdown chunking | Natural semantic boundaries | Very short sections get merged (min 200 char), loses granularity |
| Denormalized project_id/file_path in interactions | Trace queries group by file identity, not chunk_id | ~20 bytes extra per interaction row |

## Known Tech Debt

**High:**
- No log rotation for `~/.local/share/dnomia-knowledge/index.log` (launchd output)
- Stale mkdir lock (`/tmp/dnomia-knowledge.lock`) survives until reboot if hook process crashes
- `cli.py` still accesses `Store._connect()` directly (3 calls, not yet migrated to wrappers)

**Medium:**
- `install-hooks` must be re-run after adding new projects (no auto-discovery)
- Graph rebuild is full, no incremental edge updates
- Embedding model fixed to multilingual-e5-base (no per-project model config)

**Low:**
- No search result pagination (returns all results up to limit)
- Trace query grouping is case-sensitive (no normalization)
- CLI export command outputs flat CSV, no graph edge export
