"""PostgreSQL test-database isolation."""

from __future__ import annotations

from src import db


def pytest_sessionstart(session) -> None:
    db.cleanup_test_schemas()


def pytest_sessionfinish(session, exitstatus) -> None:
    db.cleanup_test_schemas()
