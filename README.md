# dnomia-knowledge

Local knowledge engine for codebases. Indexes your markdown and source code into a single SQLite database with hybrid search (FTS5 keyword + vector semantic), knowledge graph, and developer interaction tracking.

Built for [Claude Code](https://claude.ai/claude-code) via MCP (Model Context Protocol). Works entirely on your machine. Your code never leaves your computer.

## What it does

**Hybrid search** across your projects. Not just grep, not just embeddings. FTS5 (BM25) and sqlite-vec (cosine KNN) run in parallel, merged with Reciprocal Rank Fusion. Finds code and documentation that keyword search misses and vector search misranks.

**Knowledge graph** over your codebase. Chunks are connected by markdown links, shared tags, categories, import statements, and semantic similarity. Community detection (Louvain) and PageRank surface the structure of your project.

**Interaction tracking** learns what matters to you. Every file you read and edit is logged. Search results are boosted by your actual usage patterns. Trace analytics show which files are hot, which knowledge gaps exist, and which areas are decaying.

**Cross-project search** lets you query across all your indexed repositories at once. Related projects can be linked via config for unified search results.

**Continuous indexing** keeps everything fresh. Git post-commit hooks and a periodic job (launchd on macOS) re-index changed files automatically. No daemon, no persistent memory usage.

## Quick start

```bash
# Clone and install
git clone https://github.com/ceaksan/dnomia-knowledge.git
cd dnomia-knowledge
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .

# Index your first project
dnomia-knowledge index /path/to/your/project

# Search
dnomia-knowledge search "authentication middleware"

# See what files you access most
dnomia-knowledge trace hot
```

The embedding model (`intfloat/multilingual-e5-base`, ~500MB) downloads automatically on first run.

## Connect to Claude Code

Add to `~/.claude/settings.json` under `mcpServers`:

```json
{
  "dnomia-knowledge": {
    "command": "/path/to/dnomia-knowledge/.venv/bin/python",
    "args": ["-m", "dnomia_knowledge.server"],
    "env": {
      "DNOMIA_KNOWLEDGE_PROJECT": "my-project"
    }
  }
}
```

Claude Code now has access to 6 MCP tools: `search`, `index_project`, `project_info`, `graph_query`, `read_file`, and `fetch_and_index`.

### Claude Code hooks (optional)

Track file interactions automatically by adding hooks to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read|Grep",
        "hooks": [{
          "type": "command",
          "command": "/path/to/.venv/bin/python -m dnomia_knowledge.hooks.pre_tool_use"
        }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Read|Edit",
        "hooks": [{
          "type": "command",
          "command": "/path/to/.venv/bin/python -m dnomia_knowledge.hooks.post_tool_use"
        }]
      }
    ]
  }
}
```

The PreToolUse hook redirects large file reads (>300 lines) to the knowledge base search. The PostToolUse hook logs every Read/Edit for interaction tracking and search ranking.

## Project configuration

Create `.knowledge.toml` in your project root to control what gets indexed:

```toml
[project]
name = "my-project"
type = "saas"              # content | saas | static

[content]
paths = ["docs/"]
extensions = [".md", ".mdx"]

[code]
preset = "python"          # web | python | django | mixed
paths = ["src/"]
max_chunk_lines = 50

[graph]
enabled = true
edge_types = ["link", "tag", "semantic", "import"]
semantic_threshold = 0.75

[indexing]
ignore_patterns = ["node_modules", "dist", "__pycache__", ".venv"]
max_file_size_kb = 500
```

Without `.knowledge.toml`, defaults to indexing `.md` and `.mdx` files only.

## CLI reference

### Indexing

```bash
dnomia-knowledge index <path>              # Index a project directory
dnomia-knowledge index <path> --full       # Force full reindex (skip incremental)
dnomia-knowledge index-all                 # Index all registered projects
dnomia-knowledge index-all --changed       # Only projects with changes since last index
```

### Search

```bash
dnomia-knowledge search <query>                    # Search all projects
dnomia-knowledge search <query> -p my-project      # Filter by project
dnomia-knowledge search <query> -d code            # Filter by domain (code|content|all)
dnomia-knowledge search <query> --lang python      # Filter by language
```

### Trace analytics

```bash
dnomia-knowledge trace hot                 # Most accessed files (reads + edits + searches)
dnomia-knowledge trace gaps                # Searches that returned zero results
dnomia-knowledge trace decay               # Files with declining activity over time
dnomia-knowledge trace queries             # Most frequent search patterns
```

All trace commands accept `--project/-p`, `--days/-d` (default 30), and `--limit/-l` (default 20).

### Git history analysis

```bash
dnomia-knowledge git-sync <path>           # Sync git log into the database
dnomia-knowledge analyze churn             # Most modified files by insertions + deletions
dnomia-knowledge analyze hotspots          # Directory-level churn aggregation
dnomia-knowledge analyze crossover         # Fuse git churn with trace read data
```

Crossover analysis assigns signals to files based on change frequency vs read frequency:

| Signal | Meaning |
|--------|---------|
| BLIND | High churn, zero reads. Changing but never consulted. |
| TURBULENT | High churn, low reads. Unstable and under-monitored. |
| HOT | High churn, high reads. Core active area. |
| STABLE | Low churn, high reads. Settled reference code. |
| ZOMBIE | Zero churn, some reads. Read but never touched. |
| COLD | Low churn, low reads. Inactive. |

### Knowledge graph

```bash
dnomia-knowledge graph rebuild             # Rebuild all edges for a project
dnomia-knowledge graph communities         # Run Louvain community detection + PageRank
```

### Continuous indexing

```bash
dnomia-knowledge install-hooks             # Git post-commit hooks on all projects
dnomia-knowledge install-hooks --uninstall
dnomia-knowledge install-launchd           # macOS launchd job (every 5 min)
dnomia-knowledge install-launchd --uninstall
```

### Other

```bash
dnomia-knowledge project-info              # List all projects with stats
dnomia-knowledge read-file <path>          # Smart file reading with chunk awareness
dnomia-knowledge export                    # CSV export of all chunks
```

## How it works

### Search pipeline

```
Query
  -> embed with "query: " prefix (768d vector, cached)
  -> FTS5 BM25 search (keyword matching)
  -> sqlite-vec KNN search (semantic similarity)
  -> RRF merge (k=60): score = sum(1/(k + rank + 1))
  -> Fallback: prefix matching if both return empty
  -> Interaction boost: re-rank by read/edit frequency (30-day window)
  -> Return top N with snippets
```

### Indexing pipeline

```
Project directory
  -> Scan: filter by extension, size, .gitignore, config patterns
  -> For each changed file (MD5 hash comparison):
      -> .md/.mdx -> heading-based chunker (##/### splits, frontmatter)
      -> code -> Tree-sitter AST chunker (functions, classes, methods)
      -> Embed passages (batch=8, "passage: " prefix)
      -> Atomic transaction: delete old + insert chunks + insert vectors
  -> Build graph edges (link, tag, category, semantic, import)
  -> Update project metadata + git commit hash
```

### Continuous indexing

```
git commit -> post-commit hook -> file lock -> background reindex
launchd (every 5 min) -> index-all --changed -> git HEAD comparison -> reindex
```

Only one index process runs at a time. File locks prevent concurrent embedding model loads (protects 8GB RAM machines).

## Architecture

Single SQLite database with three search layers:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Keyword | FTS5 (BM25) | Porter stemmer, unicode61 tokenizer |
| Semantic | sqlite-vec | Cosine KNN on 768d normalized vectors |
| Graph | NetworkX | Louvain communities, PageRank, BFS traversal |

Embedding model: `intfloat/multilingual-e5-base` (768d). Lazy loaded on first search, auto-unloads after 10 minutes idle. Runs on 8GB RAM.

Code parsing: Tree-sitter with language pack. Extracts functions, classes, methods, structs, interfaces, enums with proper boundaries. Falls back to sliding-window chunking for unsupported languages.

### Module structure

```
src/dnomia_knowledge/
  server.py        MCP server (6 tools, thread-safe singletons)
  store.py         SQLite persistence, schema v3, migrations, triggers
  search.py        Hybrid FTS5 + vector, RRF merge, interaction boost
  indexer.py       Scan -> chunk -> embed -> store pipeline
  graph.py         Edge builder, Louvain community detection, PageRank
  embedder.py      Lazy sentence-transformer, LRU cache, auto-unload
  cli.py           Rich CLI with 10+ commands
  registry.py      .knowledge.toml config loader (Pydantic v2)
  models.py        Chunk, SearchResult, IndexResult, InteractionType
  chunker/
    md_chunker.py  Heading-based markdown splitter
    ast_chunker.py Tree-sitter AST chunker with fallback
    languages.py   Per-language AST node type mappings
  hooks/
    pre_tool_use.py   Redirects large file reads to search
    post_tool_use.py  Logs read/edit interactions
```

### Data model

| Table | Purpose |
|-------|---------|
| `projects` | Registered projects with path, type, graph config, last indexed commit |
| `chunks` | Indexed content and code pieces with metadata |
| `chunks_vec` | sqlite-vec virtual table for vector embeddings (768d) |
| `chunks_fts` | FTS5 virtual table mirroring chunk content |
| `file_index` | Per-file MD5 hash tracking for incremental indexing |
| `edges` | Knowledge graph edges (link, tag, category, semantic, import) |
| `chunk_interactions` | Read/edit/search_hit tracking for boost and analytics |
| `search_log` | Query history for gap analysis and pattern tracking |
| `git_commits` | Parsed git log entries |
| `git_file_changes` | Per-file diff stats from git history |

Triggers auto-sync FTS5 on chunk insert/update/delete. Vector cleanup triggers on chunk delete.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DNOMIA_KNOWLEDGE_DB` | `~/.local/share/dnomia-knowledge/knowledge.db` | Database path |
| `DNOMIA_KNOWLEDGE_PROJECT` | (none) | Default project for MCP search |

## Requirements

- Python 3.11+
- macOS or Linux (launchd is macOS only, git hooks work everywhere)
- ~500MB disk for embedding model (downloaded once)
- 8GB RAM minimum (embedding model loads lazily)

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v       # 276 tests
ruff check src/ tests/           # Linting
```

## License

MIT
