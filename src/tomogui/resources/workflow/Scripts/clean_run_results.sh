#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./clean_run_results.sh [--force]

Remove generated outputs from a TomoATT run folder while keeping inputs,
YAMLs, source/receiver files, and initial models.

By default this is a dry run and only prints what would be removed.
Pass --force to actually delete files.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DRY_RUN=1
if [[ "${1:-}" == "--force" ]]; then
  DRY_RUN=0
elif [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  usage >&2
  exit 2
fi

TARGET_DIRS=(
  "checkerboard_P"
  "checkerboard_S"
  "inversion_P"
  "inversion_S"
  "inversion_P_iso"
  "inversion_S_iso"
  "inversion_VpVs"
)

REMOVE_DIRS=(
  "OUTPUT_FILES"
  "img"
  "azcov_archive"
  "__pycache__"
)

REMOVE_FILE_GLOBS=(
  "tomoatt_*.out"
  "tomoatt_*.err"
  "slurm-*"
)

REMOVE_MODEL_GLOBS=(
  "2_models/model_ckb_*.h5"
  "2_models/model_final_*.h5"
)

remove_path() {
  local path="$1"
  if [[ $DRY_RUN -eq 1 ]]; then
    printf 'Would remove %s\n' "$path"
  else
    rm -rf -- "$path"
    printf 'Removed %s\n' "$path"
  fi
}

for rel_dir in "${TARGET_DIRS[@]}"; do
  dir="$ROOT_DIR/$rel_dir"
  [[ -d "$dir" ]] || continue

  for rel_remove in "${REMOVE_DIRS[@]}"; do
    path="$dir/$rel_remove"
    [[ -e "$path" ]] || continue
    remove_path "$path"
  done

  shopt -s nullglob
  for pattern in "${REMOVE_FILE_GLOBS[@]}"; do
    for path in "$dir"/$pattern; do
      remove_path "$path"
    done
  done
  for pattern in "${REMOVE_MODEL_GLOBS[@]}"; do
    for path in "$dir"/$pattern; do
      remove_path "$path"
    done
  done
  shopt -u nullglob
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run only. Re-run with --force to delete the listed paths."
fi
