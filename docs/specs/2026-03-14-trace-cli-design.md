# Trace CLI - Design Spec

CLI tool for analyzing usage patterns in dnomia-knowledge. Surfaces insights from existing `search_log` and `chunk_interactions` tables.

## Motivation

dnomia-knowledge already collects interaction data (read/edit/search_hit) and search logs. This data sits unused. Trace turns it into actionable insights: which knowledge is most referenced, where are the gaps, what is decaying.

When the project is open-sourced, trace becomes a differentiator. No existing local knowledge tool provides usage analytics.

## Prerequisites

The following fixes were applied before trace implementation (schema v2):

1. **CASCADE removal:** `chunk_interactions` table recreated without `ON DELETE CASCADE` on `chunk_id`. Interactions now survive chunk re-indexing.
2. **File-level identity:** `project_id` and `file_path` columns added to `chunk_interactions`. Trace queries group by these columns instead of `chunk_id`.
3. **Performance indexes:** Added `search_log(query)`, `search_log(result_count)`, `chunk_interactions(project_id)`, `chunk_interactions(project_id, file_path)`.
4. **Orphan cleanup:** `gc` command cleans interactions whose `chunk_id` no longer exists.

## CLI Interface

```bash
dnomia-knowledge trace <mode> [--project PROJECT] [--days DAYS] [--limit LIMIT]
```

### Common Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--project` | `-p` | None (all projects) | Filter by project ID |
| `--days` | `-d` | 30 | Time window in days (must be positive) |
| `--limit` | `-l` | 20 | Max rows to display |

**Input validation:** `--days` is validated as a positive integer at the argparse level.

**Invalid project:** If `--project` specifies a non-existent project ID, prints error and exits (same pattern as `cmd_stats`).

**Empty results:** All modes print `[dim]No trace data found.[/dim]` when the result list is empty.

### Modes

#### `trace hot`

Most interacted files. Sorted by total interaction count DESC.

**Source:** `chunk_interactions` GROUP BY `(project_id, file_path)`, LEFT JOIN `chunks` for current metadata.

**Columns:** `#`, `File`, `R` (read count), `E` (edit count), `S` (search_hit count), `Total`.

**SQL sketch:**

```sql
SELECT ci.project_id, ci.file_path,
       SUM(CASE WHEN ci.interaction = 'read' THEN 1 ELSE 0 END) as reads,
       SUM(CASE WHEN ci.interaction = 'edit' THEN 1 ELSE 0 END) as edits,
       SUM(CASE WHEN ci.interaction = 'search_hit' THEN 1 ELSE 0 END) as searches,
       COUNT(*) as total
FROM chunk_interactions ci
WHERE ci.timestamp >= datetime('now', ?)
  [AND ci.project_id = ?]
GROUP BY ci.project_id, ci.file_path
ORDER BY total DESC, ci.file_path ASC
LIMIT ?
```

#### `trace gaps`

Searches that returned 0 results. Grouped by exact query text (case-sensitive).

**Source:** `search_log` WHERE `result_count = 0`, GROUP BY `query`.

**Columns:** `#`, `Query`, `Count` (times searched), `Last Searched`.

**SQL sketch:**

```sql
SELECT query, COUNT(*) as count, MAX(timestamp) as last_searched
FROM search_log
WHERE result_count = 0
  AND timestamp >= datetime('now', ?)
  [AND project_id = ?]
GROUP BY query
ORDER BY count DESC, query ASC
LIMIT ?
```

**Known limitation:** Query grouping is case-sensitive (exact match). Semantic clustering deferred to v2.

#### `trace decay`

Files that were active in the previous period but inactive in the current period. Compares two equal time windows.

**Logic:** Given `--days N`:
- Outer bound: 2N days ago (timestamp filter for performance)
- Previous window: 2N to N days ago
- Current window: last N days
- Decay ratio: `previous_count / (current_count + 1)`
- Show files where ratio > 3

**Source:** `chunk_interactions` with two windowed COUNT queries.

**Columns:** `#`, `File`, `Before` (prev window count), `Now` (current window count), `Ratio`.

**SQL sketch:**

```sql
SELECT ci.project_id, ci.file_path,
       SUM(CASE
           WHEN ci.timestamp >= datetime('now', ?)
                AND ci.timestamp < datetime('now', ?)
           THEN 1 ELSE 0
       END) as before_count,
       SUM(CASE
           WHEN ci.timestamp >= datetime('now', ?)
           THEN 1 ELSE 0
       END) as now_count
FROM chunk_interactions ci
WHERE ci.timestamp >= datetime('now', ?)
  [AND ci.project_id = ?]
GROUP BY ci.project_id, ci.file_path
HAVING before_count > 3
   AND (before_count * 1.0 / (now_count + 1)) > 3
ORDER BY (before_count * 1.0 / (now_count + 1)) DESC,
         ci.file_path ASC
LIMIT ?
```

Parameters for `--days 30`: `-60 days` (outer bound, prev start), `-30 days` (prev end, current start), `-30 days` (current start), `-60 days` (WHERE bound).

#### `trace queries`

Most frequent search queries across all result counts.

**Source:** `search_log` GROUP BY `query`.

**Columns:** `#`, `Query`, `Count`, `Avg Results`, `Last`.

**SQL sketch:**

```sql
SELECT query, COUNT(*) as count,
       ROUND(AVG(result_count), 1) as avg_results,
       MAX(timestamp) as last_searched
FROM search_log
WHERE timestamp >= datetime('now', ?)
  [AND project_id = ?]
GROUP BY query
ORDER BY count DESC, query ASC
LIMIT ?
```

## Architecture

### Changes

**`store.py`** - 4 new methods:
- `get_hot_chunks(project_id: str | None, days: int, limit: int) -> list[dict]`
- `get_knowledge_gaps(project_id: str | None, days: int, limit: int) -> list[dict]`
- `get_decaying_chunks(project_id: str | None, days: int, limit: int) -> list[dict]`
- `get_top_queries(project_id: str | None, days: int, limit: int) -> list[dict]`

Each returns a list of dicts. Same pattern as existing `get_search_log`. Conditional SQL construction follows `get_interaction_counts` pattern (dynamic WHERE clauses).

**`cli.py`** - 1 new command function:
- `cmd_trace(args)` - Routes to store method by mode, renders Rich table

**`build_parser()`** - trace subcommand with mode positional arg.

### No New Dependencies

All logic goes into existing `store.py` (queries) and `cli.py` (rendering). No new modules, no new dependencies.

### Testing

- `test_store.py`: Unit tests for each aggregation method with seeded data
- `test_cli.py`: CLI integration tests for trace command parsing and output

## Out of Scope

- MCP tool (not needed, trace is for human insight)
- Semantic query clustering (exact match grouping is sufficient for v1)
- JSON/CSV export (can be added later with `--format` flag)
- Flow traces / session-based query chains (requires session tracking, not in current schema)
