"""Tests for CLI argument parsing."""

from __future__ import annotations

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
