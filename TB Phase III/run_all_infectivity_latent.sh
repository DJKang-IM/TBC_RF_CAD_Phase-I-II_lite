#!/usr/bin/env bash
# Run polychoric, tetrachoric, and PLS-PM infectivity analyses + HTML comparison report.
# Requires: python venv with numpy scipy pandas matplotlib (see requirements_latent_analysis.txt).
#
# Usage:
#   ./run_all_infectivity_latent.sh --csv /path/to/infectivity_complete_case_5items.csv
#   ./run_all_infectivity_latent.sh --npz /path/to/phase3.npz --meta-main /path/to/meta.csv

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$ROOT/../.venv/bin/python" ]]; then
    PY="$ROOT/../.venv/bin/python"
  else
    PY="python3"
  fi
fi

INPUT_CSV=""
INPUT_NPZ=""
META_MAIN=""
META_D5=""
BOOT=200

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv) INPUT_CSV="$2"; shift 2 ;;
    --npz) INPUT_NPZ="$2"; shift 2 ;;
    --meta-main) META_MAIN="$2"; shift 2 ;;
    --meta-d5) META_D5="$2"; shift 2 ;;
    --bootstrap) BOOT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$INPUT_CSV" && -n "$INPUT_NPZ" ]] || [[ -z "$INPUT_CSV" && -z "$INPUT_NPZ" ]]; then
  echo "Provide exactly one of --csv or --npz" >&2
  exit 2
fi

DATA_ARGS=()
if [[ -n "$INPUT_CSV" ]]; then
  DATA_ARGS=(--csv "$INPUT_CSV")
else
  DATA_ARGS=(--npz "$INPUT_NPZ")
  [[ -n "$META_MAIN" ]] && DATA_ARGS+=(--meta-main "$META_MAIN")
  [[ -n "$META_D5" ]] && DATA_ARGS+=(--meta-d5 "$META_D5")
fi

echo "=== Polychoric ==="
"$PY" analyze_infectivity_polychoric.py "${DATA_ARGS[@]}" --out artifacts/infectivity_polychoric

echo "=== Tetrachoric ==="
"$PY" analyze_infectivity_tetrachoric.py "${DATA_ARGS[@]}" --bootstrap "$BOOT" --out artifacts/infectivity_tetrachoric

echo "=== PLS-PM ==="
"$PY" analyze_infectivity_pls2block.py "${DATA_ARGS[@]}" --bootstrap "$BOOT" --out artifacts/infectivity_pls

echo "=== HTML report ==="
"$PY" generate_infectivity_methods_report.py \
  --polychoric artifacts/infectivity_polychoric \
  --tetrachoric artifacts/infectivity_tetrachoric \
  --pls artifacts/infectivity_pls \
  --out reports/infectivity_latent_methods_comparison.html

echo "Done. Open: $ROOT/reports/infectivity_latent_methods_comparison.html"
