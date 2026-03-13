"""Tests for continuous indexing features."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from dnomia_knowledge.store import Store


class TestStoreLastIndexed:
    def test_update_project_last_indexed(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        store.update_project_last_indexed("test")
        proj = store.get_project("test")
        assert proj["last_indexed"] is not None
        store.close()

    def test_update_project_last_indexed_with_commit(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        store.update_project_last_indexed("test", commit_hash="abc123")
        proj = store.get_project("test")
        assert proj["last_indexed"] is not None
        assert proj["last_indexed_commit"] == "abc123"
        store.close()

    def test_last_indexed_commit_column_exists(self, db_path):
        store = Store(db_path)
        store.register_project("test", "/tmp/test", "content")
        proj = store.get_project("test")
        assert "last_indexed_commit" in proj
        store.close()
