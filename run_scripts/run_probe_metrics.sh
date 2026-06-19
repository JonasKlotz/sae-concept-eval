#!/bin/bash
# Linear-probe upper bound: trains a logistic-regression probe per attribute and
# runs the same FBMP + TAPAScore matching on top. Not part of the main.py
# pipeline, so it is launched separately here.
#
# Portable bash (no SLURM). Wrap the python call in `srun`/`sbatch` if you
# want to run it on a cluster.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"

datasets=("cub" "coco")
models=("clip" "dinov2")

for ds in "${datasets[@]}"; do
  for mdl in "${models[@]}"; do
    echo "=== probe: dataset=${ds} model=${mdl} ==="
    python -W ignore "${PROJECT_ROOT}/src/metrics/calculate_probe_metrics.py" \
      dataset="$ds" \
      model="$mdl"
  done
done
