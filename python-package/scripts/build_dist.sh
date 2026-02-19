#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m pip install --upgrade build twine
rm -rf dist build ./*.egg-info
python -m build
python -m twine check dist/*

echo "[ok] Built and validated distribution files in: $ROOT/dist"
