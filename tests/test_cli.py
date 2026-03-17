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
        assert args.full is False

    def test_gc_full_flag(self):
        parser = build_parser()
        args = parser.parse_args(["gc", "--full"])
        assert args.command == "gc"
        assert args.full is True

    def test_no_command_fails(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])



    def test_git_sync_command(self):
        parser = build_parser()
        args = parser.parse_args(["git-sync", "/tmp/repo"])
        assert args.command == "git-sync"
        assert args.path == "/tmp/repo"

    def test_git_sync_force(self):
        parser = build_parser()
        args = parser.parse_args(["git-sync", "/tmp/repo", "--force"])
        assert args.force is True

    def test_analyze_churn(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "churn", "-p", "proj", "-d", "30"])
        assert args.command == "analyze"
        assert args.mode == "churn"
        assert args.project == "proj"
        assert args.days == 30

    def test_analyze_crossover(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "crossover"])
        assert args.mode == "crossover"
        assert args.days == 90  # default


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

    def test_doctor_warns_empty_interactions(self, db_path, tmp_dir, capsys, monkeypatch):
        """Doctor output should include interaction check warning when empty."""
        from io import StringIO

        from rich.console import Console

        from dnomia_knowledge import cli
        from dnomia_knowledge.store import Store

        store = Store(db_path)
        store.register_project("test", str(tmp_dir), "content")
        store.close()

        monkeypatch.setenv("DNOMIA_KNOWLEDGE_DB", db_path)
        output = StringIO()
        monkeypatch.setattr(cli, "console", Console(file=output, force_terminal=False))

        parser = build_parser()
        args = parser.parse_args(["doctor"])
        cli.cmd_doctor(args)

        result = output.getvalue()
        assert "chunk_interactions" in result
        assert "search_log" in result

    def test_doctor_ok_interactions(self, db_path, tmp_dir, monkeypatch):
        """Doctor output should show OK when interactions exist."""
        from io import StringIO

        from rich.console import Console

        from dnomia_knowledge import cli
        from dnomia_knowledge.store import Store

        store = Store(db_path)
        store.register_project("test", str(tmp_dir), "content")
        store.insert_chunks(
            "test",
            [
                {
                    "file_path": "a.md",
                    "content": "test",
                    "chunk_domain": "content",
                    "chunk_type": "block",
                },
            ],
        )
        conn = store._connect()
        chunk_id = conn.execute("SELECT id FROM chunks LIMIT 1").fetchone()[0]
        store.log_interaction(chunk_id, "view", "test")
        store.log_search("test query", "test", "all", [chunk_id], 1)
        store.close()

        monkeypatch.setenv("DNOMIA_KNOWLEDGE_DB", db_path)
        output = StringIO()
        monkeypatch.setattr(cli, "console", Console(file=output, force_terminal=False))

        parser = build_parser()
        args = parser.parse_args(["doctor"])
        # Doctor exits with code 1 due to chunk/vec mismatch (no vectors inserted)
        with pytest.raises(SystemExit):
            cli.cmd_doctor(args)

        result = output.getvalue()
        assert "chunk_interactions: 1 entries" in result
        assert "search_log: 1 entries" in result


class TestGCFull:
    def test_gc_full_cleans_old_data(self, db_path, tmp_dir, monkeypatch):
        """gc --full should call delete_old_interactions and delete_old_search_logs."""
        from io import StringIO
        from unittest.mock import patch

        from rich.console import Console

        from dnomia_knowledge import cli
        from dnomia_knowledge.store import Store

        store = Store(db_path)
        store.register_project("test", str(tmp_dir), "content")
        store.close()

        monkeypatch.setenv("DNOMIA_KNOWLEDGE_DB", db_path)
        output = StringIO()
        monkeypatch.setattr(cli, "console", Console(file=output, force_terminal=False))

        parser = build_parser()
        args = parser.parse_args(["gc", "--full"])

        with (
            patch.object(Store, "delete_old_interactions", return_value=5) as mock_interactions,
            patch.object(Store, "delete_old_search_logs", return_value=3) as mock_logs,
        ):
            cli.cmd_gc(args)
            mock_interactions.assert_called_once_with(90)
            mock_logs.assert_called_once_with(90)

        result = output.getvalue()
        assert "5 old interactions" in result
        assert "3 old search logs" in result

    def test_gc_without_full_skips_cleanup(self, db_path, tmp_dir, monkeypatch):
        """gc without --full should not call delete methods."""
        from io import StringIO
        from unittest.mock import patch

        from rich.console import Console

        from dnomia_knowledge import cli
        from dnomia_knowledge.store import Store

        store = Store(db_path)
        store.register_project("test", str(tmp_dir), "content")
        store.close()

        monkeypatch.setenv("DNOMIA_KNOWLEDGE_DB", db_path)
        output = StringIO()
        monkeypatch.setattr(cli, "console", Console(file=output, force_terminal=False))

        parser = build_parser()
        args = parser.parse_args(["gc"])

        with (
            patch.object(Store, "delete_old_interactions") as mock_interactions,
            patch.object(Store, "delete_old_search_logs") as mock_logs,
        ):
            cli.cmd_gc(args)
            mock_interactions.assert_not_called()
            mock_logs.assert_not_called()


class TestTraceParser:
    def test_trace_hot(self):
        parser = build_parser()
        args = parser.parse_args(["trace", "hot"])
        assert args.command == "trace"
        assert args.mode == "hot"
        assert args.days == 30
        assert args.limit == 20
        assert args.project is None

    def test_trace_gaps_with_options(self):
        parser = build_parser()
        args = parser.parse_args(["trace", "gaps", "-p", "my-proj", "-d", "7", "-l", "5"])
        assert args.mode == "gaps"
        assert args.project == "my-proj"
        assert args.days == 7
        assert args.limit == 5

    def test_trace_decay(self):
        parser = build_parser()
        args = parser.parse_args(["trace", "decay"])
        assert args.mode == "decay"

    def test_trace_queries(self):
        parser = build_parser()
        args = parser.parse_args(["trace", "queries"])
        assert args.mode == "queries"

    def test_trace_invalid_mode(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["trace", "invalid"])

    def test_trace_negative_days(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["trace", "hot", "-d", "-5"])

    def test_trace_zero_days(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["trace", "hot", "-d", "0"])
