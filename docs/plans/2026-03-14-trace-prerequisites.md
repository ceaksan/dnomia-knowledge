# Trace Prerequisites Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all issues blocking the trace CLI feature: CASCADE data loss, missing indexes, hook performance.

**Architecture:** Decouple interaction tracking from chunk lifecycle by adding `project_id`/`file_path` columns to `chunk_interactions` and removing the CASCADE FK via table recreation. Trace queries group by file-level identity (survives re-index). Search boost continues using chunk_id (current chunks only, ages out via 30-day window).

**Tech Stack:** Python, SQLite, pytest

---

## Chunk 1: Fix CASCADE Data Loss

The critical issue: `chunk_interactions.chunk_id` has `ON DELETE CASCADE`. When `indexer.py` re-indexes a file (DELETE old chunks + INSERT new), all interaction history is permanently lost.

**Fix strategy:**
1. Recreate `chunk_interactions` table without CASCADE FK (standard SQLite migration pattern)
2. Add `project_id` and `file_path` columns (denormalization)
3. Update `log_interaction`/`batch_log_interactions` to populate new columns
4. Update all callers (search.py, post_tool_use.py) in the same step
5. Update all affected tests in the same commit
6. gc command cleans up orphaned interactions

**IMPORTANT:** Steps 3-12 must all be completed before running tests. The signature changes and caller updates are atomic.

### Task 1: Write failing tests for interaction survival

**Files:**
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestInteractionSurvival:
    def test_interactions_survive_reindex(self, db_path):
        """Interactions should not be deleted when chunks are re-indexed."""
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        ids = store.insert_chunks(
            "test",
            [{"file_path": "a.md", "content": "old content", "name": "Introduction"}],
        )
        store.log_interaction(ids[0], "read", "Read", project_id="test", file_path="a.md")
        store.log_interaction(ids[0], "read", "Read", project_id="test", file_path="a.md")

        # Simulate re-index: delete old chunks, insert new
        store.delete_file_chunks("test", "a.md")
        new_ids = store.insert_chunks(
            "test",
            [{"file_path": "a.md", "content": "new content", "name": "Introduction"}],
        )

        # Old interactions should still exist (keyed by project_id + file_path)
        conn = store._connect()
        count = conn.execute(
            "SELECT COUNT(*) FROM chunk_interactions WHERE project_id = ? AND file_path = ?",
            ("test", "a.md"),
        ).fetchone()[0]
        assert count == 2

    def test_orphaned_interactions_cleaned_by_gc(self, db_path):
        """gc should remove interactions whose chunk_id no longer exists."""
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        ids = store.insert_chunks(
            "test",
            [{"file_path": "a.md", "content": "content"}],
        )
        store.log_interaction(ids[0], "read", "Read", project_id="test", file_path="a.md")
        store.delete_file_chunks("test", "a.md")

        # Interaction exists but chunk is gone
        conn = store._connect()
        count_before = conn.execute("SELECT COUNT(*) FROM chunk_interactions").fetchone()[0]
        assert count_before == 1

        cleaned = store.clean_orphaned_interactions()
        assert cleaned == 1

        count_after = conn.execute("SELECT COUNT(*) FROM chunk_interactions").fetchone()[0]
        assert count_after == 0

    def test_batch_log_interactions_with_file_info(self, db_path):
        """batch_log_interactions should accept and store project_id/file_path."""
        store, ids = _setup_project_with_chunks(db_path)
        store.batch_log_interactions([
            (ids[0], "read", "Read", "test", "file0.md"),
            (ids[1], "edit", "Edit", "test", "file1.md"),
        ])
        conn = store._connect()
        rows = conn.execute(
            "SELECT chunk_id, project_id, file_path, interaction FROM chunk_interactions ORDER BY chunk_id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["project_id"] == "test"
        assert rows[0]["file_path"] == "file0.md"
        assert rows[1]["interaction"] == "edit"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/python -m pytest tests/test_store.py::TestInteractionSurvival -v`
Expected: FAIL - `log_interaction` doesn't accept `project_id`/`file_path` params, CASCADE still active

### Task 2: Schema migration - recreate table without CASCADE

**Files:**
- Modify: `src/dnomia_knowledge/store.py:12` (SCHEMA_VERSION)
- Modify: `src/dnomia_knowledge/store.py:76-82` (table definition)
- Modify: `src/dnomia_knowledge/store.py:118-129` (indexes)
- Modify: `src/dnomia_knowledge/store.py:156-177` (_init_db migration)

- [ ] **Step 3: Update table definition in _TABLES_SQL**

```python
# FROM:
CREATE TABLE IF NOT EXISTS chunk_interactions (
    id INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    interaction TEXT NOT NULL,
    source_tool TEXT,
    timestamp TEXT DEFAULT (datetime('now'))
);

# TO:
CREATE TABLE IF NOT EXISTS chunk_interactions (
    id INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL,
    project_id TEXT,
    file_path TEXT,
    interaction TEXT NOT NULL,
    source_tool TEXT,
    timestamp TEXT DEFAULT (datetime('now'))
);
```

- [ ] **Step 4: Add indexes for new columns and search_log**

In `_INDEXES_SQL`, add after existing interaction indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_interactions_project ON chunk_interactions(project_id);
CREATE INDEX IF NOT EXISTS idx_interactions_file ON chunk_interactions(project_id, file_path);
CREATE INDEX IF NOT EXISTS idx_search_log_query ON search_log(query);
CREATE INDEX IF NOT EXISTS idx_search_log_result_count ON search_log(result_count);
```

- [ ] **Step 5: Add table recreation migration in _init_db**

In `_init_db`, after the existing `last_indexed_commit` migration, add:

```python
# Migration v2: recreate chunk_interactions without CASCADE, add project_id/file_path
has_project_col = any(
    row[1] == "project_id"
    for row in conn.execute("PRAGMA table_info(chunk_interactions)").fetchall()
)
if not has_project_col:
    conn.executescript("""
        CREATE TABLE chunk_interactions_new (
            id INTEGER PRIMARY KEY,
            chunk_id INTEGER NOT NULL,
            project_id TEXT,
            file_path TEXT,
            interaction TEXT NOT NULL,
            source_tool TEXT,
            timestamp TEXT DEFAULT (datetime('now'))
        );
        INSERT INTO chunk_interactions_new (id, chunk_id, interaction, source_tool, timestamp)
            SELECT id, chunk_id, interaction, source_tool, timestamp
            FROM chunk_interactions;
        DROP TABLE chunk_interactions;
        ALTER TABLE chunk_interactions_new RENAME TO chunk_interactions;
    """)
    # Backfill project_id/file_path from chunks table
    conn.execute("""
        UPDATE chunk_interactions SET
            project_id = (SELECT c.project_id FROM chunks c WHERE c.id = chunk_interactions.chunk_id),
            file_path = (SELECT c.file_path FROM chunks c WHERE c.id = chunk_interactions.chunk_id)
        WHERE project_id IS NULL AND chunk_id IN (SELECT id FROM chunks)
    """)
    # Clean orphaned interactions (chunks already deleted by old CASCADE)
    conn.execute(
        "DELETE FROM chunk_interactions WHERE chunk_id NOT IN (SELECT id FROM chunks)"
    )
    conn.commit()
```

- [ ] **Step 6: Bump schema version and fix version upsert**

Change `SCHEMA_VERSION = "1"` to `SCHEMA_VERSION = "2"`.

Change the version insert from `INSERT OR IGNORE` to upsert so existing DBs get updated:

```python
# FROM:
conn.execute(
    "INSERT OR IGNORE INTO system_metadata (key, value) VALUES ('schema_version', ?)",
    (SCHEMA_VERSION,),
)

# TO:
conn.execute(
    "INSERT INTO system_metadata (key, value) VALUES ('schema_version', ?) "
    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
    (SCHEMA_VERSION,),
)
```

### Task 3: Update log_interaction methods + add clean_orphaned_interactions

**Files:**
- Modify: `src/dnomia_knowledge/store.py:566-584`

- [ ] **Step 7: Update log_interaction signature**

```python
def log_interaction(
    self,
    chunk_id: int,
    interaction: str,
    source_tool: str,
    project_id: str | None = None,
    file_path: str | None = None,
) -> None:
    """Log a chunk interaction (read, edit, search_hit)."""
    conn = self._connect()
    conn.execute(
        "INSERT INTO chunk_interactions (chunk_id, project_id, file_path, interaction, source_tool) VALUES (?, ?, ?, ?, ?)",
        (chunk_id, project_id, file_path, interaction, source_tool),
    )
    conn.commit()
```

- [ ] **Step 8: Update batch_log_interactions signature**

```python
def batch_log_interactions(self, interactions: list[tuple[int, str, str, str | None, str | None]]) -> None:
    """Batch log interactions. Tuples: (chunk_id, interaction, source_tool, project_id, file_path)."""
    if not interactions:
        return
    conn = self._connect()
    conn.executemany(
        "INSERT INTO chunk_interactions (chunk_id, interaction, source_tool, project_id, file_path) VALUES (?, ?, ?, ?, ?)",
        interactions,
    )
    conn.commit()
```

- [ ] **Step 9: Add clean_orphaned_interactions method**

Add after `delete_old_interactions`:

```python
def clean_orphaned_interactions(self) -> int:
    """Remove interactions whose chunk_id no longer exists in chunks table."""
    conn = self._connect()
    cursor = conn.execute(
        "DELETE FROM chunk_interactions WHERE chunk_id NOT IN (SELECT id FROM chunks)"
    )
    conn.commit()
    return cursor.rowcount
```

### Task 4: Update all callers (atomic with Task 3)

**Files:**
- Modify: `src/dnomia_knowledge/search.py:130-135`
- Modify: `src/dnomia_knowledge/hooks/post_tool_use.py:84-86`
- Modify: `src/dnomia_knowledge/cli.py:335-348` (gc command)

- [ ] **Step 10: Update search.py _log_search_results**

```python
def _log_search_results(
    self,
    query: str,
    project_id: str | None,
    domain: str,
    results: list[SearchResult],
) -> None:
    """Log search query and mark result chunks as search_hit."""
    try:
        chunk_ids = [r.chunk_id for r in results]
        self._store.log_search(query, project_id, domain, chunk_ids, len(results))
        self._store.batch_log_interactions(
            [(r.chunk_id, "search_hit", "search", r.project_id, r.file_path) for r in results]
        )
    except Exception:
        logger.debug("Failed to log search results", exc_info=True)
```

- [ ] **Step 11: Update post_tool_use.py - batch + pass project_id/file_path**

Replace lines 84-86:

```python
# FROM:
interaction = "read" if tool_name == "Read" else "edit"
for chunk_id in chunk_ids:
    store.log_interaction(chunk_id, interaction, f"hook:{tool_name}")

# TO:
interaction = "read" if tool_name == "Read" else "edit"
batch = [
    (cid, interaction, f"hook:{tool_name}", project_id, rel_path)
    for cid in chunk_ids
]
store.batch_log_interactions(batch)
```

- [ ] **Step 12: Update gc command to clean orphaned interactions**

In `cmd_gc`, after the file cleanup loop (before `if args.full:` block), add:

```python
orphaned = store.clean_orphaned_interactions()
if orphaned:
    console.print(f"[green]Cleaned {orphaned} orphaned interactions.[/green]")
```

### Task 5: Update all affected tests

**Files:**
- Modify: `tests/test_store.py`

- [ ] **Step 13: Update test_sets_system_metadata**

```python
def test_sets_system_metadata(self, db_path):
    store = Store(db_path)
    conn = store._connect()
    row = conn.execute(
        "SELECT value FROM system_metadata WHERE key = 'schema_version'"
    ).fetchone()
    assert row is not None
    assert row[0] == "2"
```

- [ ] **Step 14: Update test_log_interaction to verify new columns**

```python
def test_log_interaction(self, db_path):
    store, ids = _setup_project_with_chunks(db_path)
    store.log_interaction(ids[0], "read", "Read", project_id="test", file_path="file0.md")
    conn = store._connect()
    row = conn.execute(
        "SELECT * FROM chunk_interactions WHERE chunk_id = ?", (ids[0],)
    ).fetchone()
    assert row is not None
    assert row["interaction"] == "read"
    assert row["source_tool"] == "Read"
    assert row["project_id"] == "test"
    assert row["file_path"] == "file0.md"

def test_log_interaction_without_file_info(self, db_path):
    """log_interaction should work without project_id/file_path (backward compat)."""
    store, ids = _setup_project_with_chunks(db_path)
    store.log_interaction(ids[0], "read", "Read")
    conn = store._connect()
    row = conn.execute(
        "SELECT * FROM chunk_interactions WHERE chunk_id = ?", (ids[0],)
    ).fetchone()
    assert row is not None
    assert row["project_id"] is None
    assert row["file_path"] is None
```

- [ ] **Step 15: Run full test suite**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 16: Lint check**

Run: `cd /Users/ceair/Desktop/DNM_Projects/dnomia-knowledge && .venv/bin/ruff check src/ tests/`
Expected: No errors

- [ ] **Step 17: Commit**

```bash
git add src/dnomia_knowledge/store.py src/dnomia_knowledge/search.py src/dnomia_knowledge/hooks/post_tool_use.py src/dnomia_knowledge/cli.py tests/test_store.py
git commit -m "fix: decouple interaction tracking from chunk lifecycle

Recreate chunk_interactions table without CASCADE FK. Add project_id and
file_path columns so interactions survive re-indexing. Update all callers
to pass file-level identity. Batch hook writes for performance.
gc cleans orphaned interactions. Schema version bumped to 2."
```

---

## Chunk 2: Update Spec Document

### Task 6: Update trace spec with court findings

**Files:**
- Modify: `docs/specs/2026-03-14-trace-cli-design.md`

- [ ] **Step 18: Update spec**

Add/change in the spec:
1. Add "Prerequisites" section documenting the CASCADE fix (done in Chunk 1)
2. Update `trace hot` SQL to group by `(ci.project_id, ci.file_path)` and JOIN chunks for current metadata
3. Update `trace decay` SQL to include outer timestamp bound (`WHERE ci.timestamp >= datetime('now', ?)` with `-2N days`)
4. Add secondary sort keys for stability (e.g., `ORDER BY total DESC, ci.file_path ASC`)
5. Add empty result handling: print `[dim]No trace data found. Run searches or use indexed projects to generate usage data.[/dim]`
6. Add input validation: `--days` validated as positive integer at argparse level
7. Add known limitation: query grouping is case-sensitive (exact match), semantic clustering deferred to v2
8. Document that `--project` with invalid ID prints error and exits (same pattern as `cmd_stats`)
9. Remove "No New Files" claim from Architecture section (was invalidated by court, though ultimately no new files were needed for the prereqs)

- [ ] **Step 19: Commit**

```bash
git add docs/specs/2026-03-14-trace-cli-design.md
git commit -m "docs: update trace spec with court findings and prerequisites"
```

---

## Summary

| Chunk | Tasks | Files | Risk |
|-------|-------|-------|------|
| 1: CASCADE fix | 1-5 | store.py, search.py, post_tool_use.py, cli.py, test_store.py | High - schema migration, all callers |
| 2: Spec update | 6 | trace-cli-design.md | Low - docs only |

**Total:** 2 commits, 6 files modified, 0 new files created.

**After this plan:** Trace feature can be implemented cleanly on top of reliable interaction data. Create a second plan for the trace CLI implementation itself.
