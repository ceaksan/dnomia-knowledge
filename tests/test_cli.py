"""Tests for CLI argument parsing."""

from __future__ import annotations

import os

import pytest

from dnomia_knowledge.cli import build_parser


class TestCLIParser:
    def test_index_command(self):
        parser = build_parser()
        args = parser.parse_args(["index", "/tmp/project"])
        assert args.command == "index"
        assert args.path == "/tmp/project"
        assert args.full is False

    def test_index_full(self):
        parser = build_parser()
        args = parser.parse_args(["index", "/tmp/project", "--full"])
        assert args.full is True

    def test_search_command(self):
        parser = build_parser()
        args = parser.parse_args(["search", "auth middleware"])
        assert args.command == "search"
        assert args.query == "auth middleware"
        assert args.limit == 10
        assert args.domain == "all"

    def test_search_with_options(self):
        parser = build_parser()
        args = parser.parse_args(["search", "query", "-p", "proj", "-d", "code", "-l", "5"])
        assert args.project == "proj"
        assert args.domain == "code"
        assert args.limit == 5

    def test_projects_command(self):
        parser = build_parser()
        args = parser.parse_args(["projects"])
        assert args.command == "projects"

    def test_stats_command(self):
        parser = build_parser()
        args = parser.parse_args(["stats", "my-project"])
        assert args.command == "stats"
        assert args.project_id == "my-project"

    def test_doctor_no_project(self):
        parser = build_parser()
        args = parser.parse_args(["doctor"])
        assert args.command == "doctor"
        assert args.project_id is None

    def test_doctor_with_project(self):
        parser = build_parser()
        args = parser.parse_args(["doctor", "my-project"])
        assert args.project_id == "my-project"

    def test_gc_command(self):
        parser = build_parser()
        args = parser.parse_args(["gc"])
        assert args.command == "gc"

    def test_no_command_fails(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestDoctorChecks:
    def test_chunk_vec_mismatch_detected(self, db_path):
        """Doctor should detect chunk/vec count mismatch."""
        from dnomia_knowledge.store import Store

        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        # Insert chunk without vector - creates mismatch
        store.insert_chunks(
            "test",
            [
                {
                    "file_path": "a.md",
                    "content": "test content",
                    "chunk_domain": "content",
                    "chunk_type": "block",
                },
            ],
        )
        conn = store._connect()
        chunk_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE project_id = 'test'"
        ).fetchone()[0]
        vec_count = conn.execute(
            "SELECT COUNT(*) FROM chunks_vec cv "
            "JOIN chunks c ON cv.id = c.id "
            "WHERE c.project_id = 'test'"
        ).fetchone()[0]
        assert chunk_count == 1
        assert vec_count == 0  # mismatch detected
        store.close()

    def test_missing_file_detected(self, db_path, tmp_dir):
        """Doctor should detect files in index that don't exist on disk."""
        from dnomia_knowledge.store import Store

        store = Store(db_path)
        store.register_project("test", str(tmp_dir), "content")
        store.upsert_file_index("test", "gone.md", "somehash", 1)
        # gone.md doesn't exist on disk
        hashes = store.get_all_file_hashes("test")
        missing = [rp for rp in hashes if not os.path.exists(os.path.join(str(tmp_dir), rp))]
        assert len(missing) == 1
        assert "gone.md" in missing
        store.close()
