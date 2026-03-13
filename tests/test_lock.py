"""Tests for file lock utility."""

from __future__ import annotations

import os
import tempfile

import pytest

from dnomia_knowledge.lock import IndexLock


class TestIndexLock:
    def test_acquire_and_release(self, tmp_dir):
        lock_path = str(tmp_dir / "test.lock")
        lock = IndexLock(lock_path)
        assert lock.acquire() is True
        lock.release()

    def test_non_blocking_fails_when_held(self, tmp_dir):
        lock_path = str(tmp_dir / "test.lock")
        lock1 = IndexLock(lock_path)
        lock2 = IndexLock(lock_path)
        assert lock1.acquire() is True
        assert lock2.acquire() is False
        lock1.release()

    def test_acquire_after_release(self, tmp_dir):
        lock_path = str(tmp_dir / "test.lock")
        lock1 = IndexLock(lock_path)
        lock2 = IndexLock(lock_path)
        assert lock1.acquire() is True
        lock1.release()
        assert lock2.acquire() is True
        lock2.release()

    def test_context_manager(self, tmp_dir):
        lock_path = str(tmp_dir / "test.lock")
        with IndexLock(lock_path) as acquired:
            assert acquired is True

    def test_context_manager_skip_when_held(self, tmp_dir):
        lock_path = str(tmp_dir / "test.lock")
        lock1 = IndexLock(lock_path)
        assert lock1.acquire() is True
        with IndexLock(lock_path) as acquired:
            assert acquired is False
        lock1.release()
