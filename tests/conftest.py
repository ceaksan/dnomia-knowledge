"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def db_path(tmp_dir):
    """Temporary database path."""
    return str(tmp_dir / "test.db")


@pytest.fixture
def sample_markdown():
    """Sample markdown content for testing."""
    return """---
title: Test Article
tags: [python, testing]
categories: [tutorial]
description: A test article about Python testing
---

## Introduction

This is a test article about Python testing. It covers the basics of pytest.

## Writing Tests

Here is how you write a test:

```python
def test_example():
    assert 1 + 1 == 2
```

Always use descriptive test names.

## Running Tests

You can run tests with:

```bash
pytest -v
```

## Conclusion

Testing is important for code quality.
"""


@pytest.fixture
def sample_markdown_file(tmp_dir, sample_markdown):
    """Write sample markdown to a file and return the path."""
    p = tmp_dir / "test-article.md"
    p.write_text(sample_markdown)
    return p
