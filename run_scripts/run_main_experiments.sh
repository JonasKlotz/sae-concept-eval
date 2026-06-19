#!/bin/bash
# Full experiment sweep: embed -> train SAE -> all metrics (CKNNA, FMS,
# monosemanticity, GT matching [F1, FBMP, nnomp], TAPAScore, visualizations).
# Everything is driven by src/main.py; embedding and training are skipped
# automatically if their outputs already exist.
#
# Portable bash (no SLURM). Wrap the python call in `srun`/`sbatch` if you
# want to run it on a cluster.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"

datasets=("cub" "coco")
models=("clip" "dinov2")
saes=("topk" "batchtopk" "matryoshka" "jumprelu" "random" "frozen")
dict_sizes=(128 256 512 1024 2048 4096)

for ds in "${datasets[@]}"; do
  for mdl in "${models[@]}"; do
    for s in "${saes[@]}"; do
      for dict_size in "${dict_sizes[@]}"; do
        echo "=== dataset=${ds} model=${mdl} sae=${s} dict_size=${dict_size} ==="
        python -W ignore "${PROJECT_ROOT}/src/main.py" \
          dataset="$ds" \
          model="$mdl" \
          sae="$s" \
          sae.dict_size="$dict_size"
      done
    done
  done
done
