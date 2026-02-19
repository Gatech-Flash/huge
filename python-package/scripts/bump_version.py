#!/usr/bin/env python3
"""Bump pyhuge version in project metadata files."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
INIT_PY = ROOT / "pyhuge" / "__init__.py"
CHANGELOG = ROOT / "CHANGELOG.md"


def _validate_version(version: str) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Version must follow MAJOR.MINOR.PATCH (e.g. 0.2.0).")


def _replace_regex(path: pathlib.Path, pattern: str, repl: str) -> bool:
    text = path.read_text(encoding="utf-8")
    new_text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n == 0:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def _ensure_changelog_heading(version: str) -> None:
    text = CHANGELOG.read_text(encoding="utf-8")
    heading = f"## {version}"
    if heading in text:
        return
    lines = text.splitlines()
    insert_at = 1 if lines and lines[0].startswith("# ") else 0
    block = ["", heading, "", "- TBD", ""]
    lines[insert_at:insert_at] = block
    CHANGELOG.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bump pyhuge version.")
    parser.add_argument("version", help="New version (MAJOR.MINOR.PATCH)")
    args = parser.parse_args()

    try:
        _validate_version(args.version)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    ok_pyproject = _replace_regex(
        PYPROJECT,
        r'(^version\s*=\s*")([^"]+)(")',
        rf'\g<1>{args.version}\g<3>',
    )
    ok_init = _replace_regex(
        INIT_PY,
        r'(^__version__\s*=\s*")([^"]+)(")',
        rf'\g<1>{args.version}\g<3>',
    )
    if not ok_pyproject or not ok_init:
        print("error: failed to update version fields.", file=sys.stderr)
        return 1

    _ensure_changelog_heading(args.version)
    print(f"Updated version to {args.version}")
    print(f"- {PYPROJECT}")
    print(f"- {INIT_PY}")
    print(f"- {CHANGELOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
