"""File lock utility for preventing concurrent index runs."""

from __future__ import annotations

import fcntl
import os

LOCK_PATH = "/tmp/dnomia-knowledge.lockfile"


class IndexLock:
    """Non-blocking file lock using fcntl."""

    def __init__(self, path: str = LOCK_PATH):
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Try to acquire lock. Returns True if acquired, False if held by another process."""
        try:
            self._fd = os.open(self.path, os.O_CREAT | os.O_WRONLY)
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            return False

    def release(self) -> None:
        """Release the lock."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *args) -> None:
        self.release()
