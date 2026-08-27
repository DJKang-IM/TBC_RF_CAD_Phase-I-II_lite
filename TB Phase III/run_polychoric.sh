#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-${ROOT}/../.venv/bin/python}}"
[[ -x "$PY" ]] || PY=python3
exec "$PY" "$ROOT/analyze_infectivity_polychoric.py" "$@" --out "${OUT:-artifacts/infectivity_polychoric}"
