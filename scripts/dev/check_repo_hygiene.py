#!/usr/bin/env python3
"""Repository hygiene checks for tracked files.

Checks:
1) No tracked cache/noise files (e.g., .pyc, .DS_Store, __pycache__).
2) No hardcoded local absolute paths in Python sources.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

_NOISE_PATTERNS = (
    re.compile(r"(^|/)\.DS_Store$"),
    re.compile(r"(^|/)__pycache__(/|$)"),
    re.compile(r"\.py[co]$"),
)

_SOURCE_DIR_PREFIXES = ("cortexnet/", "scripts/", "examples/", "tests/")
_SOURCE_SUFFIXES = (".py",)
_SELF_PATH = "scripts/dev/check_repo_hygiene.py"
_LOCAL_ABS_PATH_PATTERNS = (
    re.compile(r"/Users/[^\"' \n\t]+"),
    re.compile(r"C:\\\\Users\\\\[^\"' \n\t]+"),
)


def _tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [line.strip() for line in out.splitlines() if line.strip()]


def _check_noise_files(tracked: list[str]) -> list[str]:
    bad = []
    for path in tracked:
        if any(p.search(path) for p in _NOISE_PATTERNS):
            bad.append(path)
    return bad


def _check_local_paths(tracked: list[str]) -> list[str]:
    findings: list[str] = []
    for rel in tracked:
        if rel == _SELF_PATH:
            continue
        if not rel.endswith(_SOURCE_SUFFIXES):
            continue
        if not rel.startswith(_SOURCE_DIR_PREFIXES):
            continue
        abs_path = ROOT / rel
        try:
            text = abs_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for patt in _LOCAL_ABS_PATH_PATTERNS:
            for m in patt.finditer(text):
                findings.append(f"{rel}: {m.group(0)}")
    return findings


def main() -> int:
    tracked = _tracked_files()

    bad_noise = _check_noise_files(tracked)
    bad_local = _check_local_paths(tracked)

    if not bad_noise and not bad_local:
        print("repo_hygiene_ok")
        return 0

    if bad_noise:
        print("Tracked noise/cache files detected:")
        for p in bad_noise:
            print(f"  - {p}")

    if bad_local:
        print("Hardcoded local absolute paths detected:")
        for line in bad_local:
            print(f"  - {line}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
