"""Explicit privileged PostgreSQL bootstrap for Research Harness."""

from __future__ import annotations

import os
import re

import psycopg
from psycopg import sql


def _required(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        raise SystemExit(f"{name} must be set")
    return value


def _identifier(name: str, value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", value):
        raise SystemExit(f"{name} must be a PostgreSQL identifier")
    return value


def main() -> None:
    admin_url = _required("POSTGRES_ADMIN_URL")
    database = _identifier("RESEARCH_HARNESS_DB_NAME", os.getenv("RESEARCH_HARNESS_DB_NAME", "research_harness"))
    username = _identifier("RESEARCH_HARNESS_DB_USER", os.getenv("RESEARCH_HARNESS_DB_USER", "research_harness"))
    password = _required("RESEARCH_HARNESS_DB_PASSWORD")

    with psycopg.connect(admin_url, autocommit=True) as connection:
        role = connection.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (username,)).fetchone()
        if role is None:
            connection.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(username), sql.Literal(password)))

        exists = connection.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,)).fetchone()
        if exists is None:
            connection.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(sql.Identifier(database), sql.Identifier(username)))

    print(f"PostgreSQL role and database are ready: {username}@{database}")


if __name__ == "__main__":
    main()
