#!/bin/bash
# Synthetic-data sanity check: trains a classifier on the synCUB / synCOCO paired
# images to verify the targeted attribute actually changes between each pair.
# Not part of the main.py pipeline, so it is launched separately here.
#
# Portable bash (no SLURM). Wrap the python call in `srun`/`sbatch` if you
# want to run it on a cluster.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"

datasets=("cub" "coco")

for ds in "${datasets[@]}"; do
  echo "=== classifier: dataset=${ds} ==="
  python "${PROJECT_ROOT}/src/scripts/classifier_for_synthetic_data_evaluation.py" \
    dataset="$ds"
done
