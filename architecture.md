# dnomia-knowledge - Architecture

Unified knowledge management MCP server. Indexes markdown and code files with hybrid search (FTS5 keyword + vector semantic) and knowledge graph over a single SQLite database.

<!--
Living Architecture Template v1.0
Source: https://github.com/ceaksan/living-architecture
Depth: L2
Last verified: 2026-03-14
-->

## Stack & Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| mcp[cli] | >=1.13.0 | FastMCP server framework |
| sentence-transformers | >=3.0.0 | intfloat/multilingual-e5-base embeddings (768d) |
| sqlite-vec | >=0.1.6 | Vector KNN search extension for SQLite |
| tree-sitter | >=0.24.0 | AST parsing for code chunking |
| tree-sitter-language-pack | >=0.7.0 | Language grammars (Python, JS/TS, Go, Rust, etc.) |
| networkx | >=3.0 | Knowledge graph (Louvain communities, PageRank) |
| pydantic | >=2.0.0 | Config validation (.knowledge.toml schema) |
| python-frontmatter | >=1.0.0 | YAML frontmatter extraction from markdown |
| pathspec | >=0.12.0 | .gitignore pattern matching |
| rich | >=13.0.0 | CLI tables and progress bars |
| pyyaml | >=6.0 | TOML/YAML config parsing |

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
├── server.py          # MCP server entry point, 6 tools (search, index, graph, read, fetch)
├── store.py           # SQLite persistence, schema, CRUD, triggers, migrations
├── embedder.py        # Lazy sentence-transformer, query/passage prefix, LRU cache
├── indexer.py         # Scan → filter → chunk → embed → store pipeline
├── search.py          # Hybrid FTS5+vector, RRF merge, interaction boost, cross-project
├── graph.py           # Edge builder (link/tag/category/semantic/import), Louvain, PageRank
├── lock.py            # fcntl file lock for preventing concurrent index runs
├── cli.py             # Rich CLI: index, search, graph, install-hooks, install-launchd
├── registry.py        # .knowledge.toml config loader, Pydantic v2 schema
├── presets.py         # Extension presets (web, python, django, mixed)
├── models.py          # Shared dataclasses (Chunk, SearchResult, IndexResult)
├── chunker/
│   ├── base.py        # Chunker protocol definition
│   ├── md_chunker.py  # Heading-based markdown splitter (##/###, frontmatter, min 200 char)
│   ├── ast_chunker.py # Tree-sitter AST chunker with sliding-window fallback
│   └── language.py    # Per-language AST node type configs
└── hooks/
    ├── pre_tool_use.py  # Blocks Read/Grep on indexed large files (>300 lines)
    └── post_tool_use.py # Logs read/edit interactions for search ranking boost
```

```
tests/
├── conftest.py           # Shared fixtures (tmp_dir, store, embedder, indexer)
├── test_store.py         # SQLite ops, triggers, indexes
├── test_embedder.py      # Lazy loading, caching, batching
├── test_chunker.py       # Heading-based markdown splitting
├── test_ast_chunker.py   # AST extraction, language support
├── test_indexer.py       # File scanning, incremental detection
├── test_search.py        # FTS5, vector, RRF merge, interaction boost
├── test_graph.py         # Edge building, Louvain, PageRank, BFS
├── test_server.py        # MCP tool interface
├── test_cli.py           # CLI command handlers
├── test_hooks.py         # Pre/post tool use hooks
├── test_integration.py   # End-to-end flows
└── test_continuous.py    # Git hooks, launchd, change detection
```

## Data Flow

### Indexing

```
Project directory
  → Scan: filter binary/size/ignore (.gitignore + config patterns)
  → For each changed file (MD5 hash comparison):
      → Route by extension: .md/.mdx → MdChunker | code → AstChunker
      → Chunk list (name, type, lines, content, metadata)
      → Embed passages (batch=8, "passage: " prefix)
      → Transaction: delete old → insert chunks → insert vectors → update file_index
  → Build graph edges (if enabled)
  → Update projects.last_indexed + last_indexed_commit
```

### Search

```
Query text
  → embed_query() (cached, "query: " prefix, 768d vector)
  → Parallel:
      FTS5 BM25 search → ranked list A
      sqlite-vec KNN   → ranked list B
  → RRF merge (k=60): score = sum(1/(k + rank + 1))
  → If empty: retry FTS with prefix matching (word*)
  → Interaction boost: re-rank by read/edit frequency (30-day window)
  → Return top N with snippets
```

### Continuous Indexing

```
git commit → post-commit hook → mkdir lock → dnomia-knowledge index . (background)
launchd (5min) → index-all --changed → fcntl lock → git HEAD comparison → re-index changed
```

## Route / API Structure

MCP tools (stdio transport):

```
search(query, domain, project, limit, language, file_pattern)
  # Hybrid FTS5+vector search, returns ranked results with snippets

index_project(path, incremental)
  # Index directory, auto-detects .knowledge.toml config

project_info(project)
  # List projects with chunk/file counts, graph status, last indexed

graph_query(project, mode, chunk_id, depth)
  # mode=neighbors: BFS traversal | mode=communities: Louvain clusters

read_file(path, query)
  # Smart file reader, large files return chunk metadata + first 50 lines

fetch_and_index(url, project)
  # HTTP fetch → strip HTML → chunk → embed → searchable immediately
```

CLI commands:

```
dnomia-knowledge index <path>              # Index project directory
dnomia-knowledge search <query>            # Search with Rich table output
dnomia-knowledge project-info              # List all projects
dnomia-knowledge graph rebuild|communities # Graph operations
dnomia-knowledge read-file <path>          # Smart file reading
dnomia-knowledge index-all [--changed]     # Index all registered projects
dnomia-knowledge install-hooks             # Git post-commit hooks
dnomia-knowledge install-launchd           # macOS periodic job (5min)
dnomia-knowledge export                    # CSV export of chunks
```

## Data Model

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `projects` | Registered project metadata | id, path, type, graph_enabled, last_indexed, last_indexed_commit, config_hash |
| `chunks` | Indexed content/code pieces | id, project_id, file_path, chunk_domain, chunk_type, name, language, start_line, end_line, content, metadata (JSON) |
| `chunks_vec` | Vector embeddings (sqlite-vec virtual) | id, embedding (768d float32) |
| `chunks_fts` | Full-text index (FTS5 virtual) | content mirror, BM25 ranking |
| `file_index` | Per-file tracking for incremental indexing | project_id, file_path, file_hash (MD5), chunk_count, last_indexed |
| `edges` | Knowledge graph relationships | source_id, target_id, edge_type (link/tag/category/semantic/import), weight, metadata |
| `chunk_interactions` | Usage tracking for search ranking | chunk_id, interaction (read/edit/search_hit), source_tool, timestamp |
| `search_log` | Query analytics | query, project_id, domain, result_chunk_ids, result_count, timestamp |
| `system_metadata` | Global config | embedding_model, embedding_dim, schema_version |

**Triggers:**
- `chunks INSERT` → auto-insert to `chunks_fts`
- `chunks UPDATE` → auto-update `chunks_fts`
- `chunks DELETE` → auto-delete from `chunks_fts` + `chunks_vec`

**Indexes:**
- `chunks(project_id, file_path)` for file-level lookups
- `file_index(project_id, file_path)` UNIQUE for incremental detection
- `edges(source_id)`, `edges(target_id)` for graph traversal

## Configuration & Environment

| Variable | Purpose | Default |
|----------|---------|---------|
| `DNOMIA_KNOWLEDGE_DB` | SQLite database file path | `~/.local/share/dnomia-knowledge/knowledge.db` |
| `DNOMIA_KNOWLEDGE_PROJECT` | Default project ID for MCP search tool | (none) |

### Project Config (.knowledge.toml)

Optional per-project config in project root:

```toml
[project]
name = "my-project"
type = "saas"          # content | saas | static

[content]
paths = ["docs/"]
extensions = [".md", ".mdx"]
chunking = "heading"   # heading | sliding

[code]
preset = "python"      # web | python | django | mixed
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

- File access restricted to registered project paths (read_file tool validates)
- No network access except explicit fetch_and_index (user-initiated URL fetch)
- SQLite WAL mode prevents corruption from concurrent reads
- fcntl/mkdir file locks prevent concurrent writes (OOM protection)
- No authentication layer (local-only MCP server, stdio transport)
- Hooks never crash Claude Code session (try/except with silent failures)

## Constraints & Trade-offs

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

## Known Tech Debt

**High:**
- No log rotation for `~/.local/share/dnomia-knowledge/index.log` (launchd output)
- Stale mkdir lock (`/tmp/dnomia-knowledge.lock`) survives until reboot if hook process crashes

**Medium:**
- `install-hooks` must be re-run after adding new projects (no auto-discovery)
- Graph rebuild is full rebuild, no incremental edge updates
- Embedding model fixed to multilingual-e5-base (no per-project model config)

**Low:**
- `chunks_vec` cleanup relies on SQLite trigger (no explicit vector garbage collection)
- CLI export command outputs flat CSV, no graph edge export
- No search result pagination (returns all results up to limit)

## Code Hotspots

| File | Changes | Risk |
|------|---------|------|
| `store.py` | Schema changes, new columns, migrations | High: all data persistence flows through here |
| `indexer.py` | Pipeline changes, new file types, graph integration | High: core indexing logic, transaction boundaries |
| `cli.py` | New commands (continuous indexing added 3 commands) | Medium: grows with features, Rich formatting |
| `search.py` | Ranking changes, new filters, cross-project | Medium: search quality directly affected |
| `ast_chunker.py` | New language support, node type configs | Low: well-isolated, language.py handles config |

---

## Optional Modules

### Background Jobs

| Job | Trigger | Purpose |
|-----|---------|---------|
| git post-commit hook | Every `git commit` | Background re-index of committed project |
| launchd periodic job | Every 5 minutes | `index-all --changed` for all registered projects |

**Concurrency:** Single-run guaranteed via file locks. Hook uses `mkdir /tmp/dnomia-knowledge.lock` (atomic POSIX). CLI uses `fcntl.flock` on `/tmp/dnomia-knowledge.lockfile`. If lock held, job exits silently (code 0).

**Change detection:** Git HEAD hash comparison (`last_indexed_commit` column). Non-git directories fall back to recursive mtime scan.

**Installation:**
```bash
dnomia-knowledge install-hooks      # git post-commit hooks on all projects
dnomia-knowledge install-launchd    # macOS launchd periodic job
```

### Search

Core functionality of this project. See Data Flow and Route/API sections above.

| Component | Technology | Detail |
|-----------|-----------|--------|
| Keyword search | FTS5 | BM25 ranking, porter stemmer, unicode61 tokenizer |
| Semantic search | sqlite-vec | Cosine KNN on 768d normalized vectors |
| Merge strategy | RRF | Reciprocal Rank Fusion (k=60) |
| Fallback | FTS5 prefix | word* matching when both searches return empty |
| Re-ranking | Interaction boost | read/edit frequency from last 30 days |
| Cross-project | search_cross() | Multi-project search with per-project score normalization |

### Dependency Graph

Internal module dependency flow:

```
server.py → store.py, embedder.py, indexer.py, search.py
indexer.py → store.py, embedder.py, chunker/, graph.py, registry.py, lock.py
search.py → store.py, embedder.py
graph.py → store.py
cli.py → store.py, embedder.py, indexer.py, search.py, lock.py
registry.py → presets.py, models.py
chunker/ast_chunker.py → chunker/language.py, chunker/base.py
chunker/md_chunker.py → chunker/base.py
hooks/ → store.py (lightweight, no embedder dependency)
```

No circular dependencies. `store.py` is the leaf dependency (depends on nothing internal). `embedder.py` is independent. All other modules compose upward.
