"""Tests for git log parser and sync engine."""

from __future__ import annotations

from dnomia_knowledge.git_sync import parse_git_log_output, parse_numstat_line


class TestParseNumstatLine:
    def test_normal_file(self):
        result = parse_numstat_line("10\t5\tsrc/main.py")
        assert result == {
            "file_path": "src/main.py",
            "old_file_path": None,
            "insertions": 10,
            "deletions": 5,
            "change_type": "M",
            "is_binary": 0,
        }

    def test_new_file(self):
        # numstat can't distinguish A from M; parser returns M
        result = parse_numstat_line("25\t0\tsrc/new.py")
        assert result == {
            "file_path": "src/new.py",
            "old_file_path": None,
            "insertions": 25,
            "deletions": 0,
            "change_type": "M",
            "is_binary": 0,
        }

    def test_binary_file(self):
        result = parse_numstat_line("-\t-\timage.png")
        assert result == {
            "file_path": "image.png",
            "old_file_path": None,
            "insertions": None,
            "deletions": None,
            "change_type": "M",
            "is_binary": 1,
        }

    def test_rename(self):
        result = parse_numstat_line("0\t0\told.py => new.py")
        assert result["file_path"] == "new.py"
        assert result["old_file_path"] == "old.py"
        assert result["change_type"] == "R"

    def test_brace_rename(self):
        result = parse_numstat_line("3\t1\tsrc/{old => new}/file.py")
        assert result["file_path"] == "src/new/file.py"
        assert result["old_file_path"] == "src/old/file.py"
        assert result["change_type"] == "R"

    def test_empty_line(self):
        result = parse_numstat_line("")
        assert result is None

    def test_whitespace_only(self):
        result = parse_numstat_line("   ")
        assert result is None

    def test_zero_changes(self):
        result = parse_numstat_line("0\t0\tsrc/empty.py")
        assert result == {
            "file_path": "src/empty.py",
            "old_file_path": None,
            "insertions": 0,
            "deletions": 0,
            "change_type": "M",
            "is_binary": 0,
        }


class TestParseGitLogOutput:
    def test_single_commit(self):
        raw = (
            "abc123def456789012345678901234567890abcd\x001700000000\x00feat: add thing\n"
            "10\t5\tsrc/main.py\n"
            "3\t1\tsrc/util.py"
        )
        commits, changes = parse_git_log_output(raw, "test-project")
        assert len(commits) == 1
        assert commits[0]["hash"] == "abc123def456789012345678901234567890abcd"
        assert commits[0]["timestamp"] == 1700000000
        assert commits[0]["summary"] == "feat: add thing"
        assert len(changes) == 2

    def test_multiple_commits(self):
        raw = (
            "abc123def456789012345678901234567890abcd\x001700000000\x00first\n"
            "10\t5\tfile1.py\n"
            "\n"
            "def4567890123456789012345678901234567890\x001700086400\x00second\n"
            "3\t1\tfile2.py"
        )
        commits, changes = parse_git_log_output(raw, "test-project")
        assert len(commits) == 2
        assert len(changes) == 2

    def test_empty_output(self):
        commits, changes = parse_git_log_output("", "test-project")
        assert commits == []
        assert changes == []

    def test_commit_with_binary_file(self):
        raw = "abc123def456789012345678901234567890abcd\x001700000000\x00add image\n-\t-\tlogo.png"
        commits, changes = parse_git_log_output(raw, "test-project")
        assert len(commits) == 1
        assert len(changes) == 1
        assert changes[0]["is_binary"] == 1

    def test_project_id_set(self):
        raw = "abc123def456789012345678901234567890abcd\x001700000000\x00init\n1\t0\tREADME.md"
        commits, changes = parse_git_log_output(raw, "my-project")
        assert commits[0]["project_id"] == "my-project"
        assert changes[0]["project_id"] == "my-project"


class TestValidateRef:
    def test_valid_full_sha(self):
        from dnomia_knowledge.git_sync import validate_ref

        assert validate_ref("abc123def456789012345678901234567890abcd") is True

    def test_valid_short_sha(self):
        from dnomia_knowledge.git_sync import validate_ref

        assert validate_ref("abc1") is True

    def test_too_short(self):
        from dnomia_knowledge.git_sync import validate_ref

        assert validate_ref("abc") is False

    def test_invalid_chars(self):
        from dnomia_knowledge.git_sync import validate_ref

        assert validate_ref("xyz1234") is False

    def test_dash_rejected(self):
        from dnomia_knowledge.git_sync import validate_ref

        assert validate_ref("--option") is False

    def test_empty(self):
        from dnomia_knowledge.git_sync import validate_ref

        assert validate_ref("") is False
