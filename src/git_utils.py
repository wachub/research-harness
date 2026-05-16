"""Small Git helpers for recording experiment provenance."""

from __future__ import annotations

import subprocess


def get_current_commit_hash() -> str | None:
    """Return the current Git commit hash, or None if unavailable."""

    result = _run_git(["rev-parse", "HEAD"])
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def get_working_tree_status() -> str:
    """Return `git status --short`, or a readable error string."""

    result = _run_git(["status", "--short"])
    if result.returncode != 0:
        return result.stderr.strip() or "git status unavailable"
    return result.stdout


def is_working_tree_clean() -> bool:
    """Return True when Git is available and the working tree has no changes."""

    result = _run_git(["status", "--short"])
    return result.returncode == 0 and result.stdout.strip() == ""


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(["git", *args], 1, "", str(exc))

