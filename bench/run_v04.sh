#!/bin/bash
#SBATCH -A m4895
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 128
#SBATCH -J almond_v04
#SBATCH -o logs/%x_%j.out

# v0.4 benchmark on a dedicated A100 node: single-transform all modes vs
# ducc0 @64t, plus batched per-column all modes vs ducc0 ntrans @64t.

set -e
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate simaster

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/.."
mkdir -p bench/logs results

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "cores: $(nproc)"

echo "=== single-transform, all modes ==="
python bench/bench_modes.py --nsides 128 256 512 1024 2048 \
    --ducc-threads 64 --tag v04node

for spec in "128 64" "256 64" "512 32" "1024 16" "2048 8"; do
    set -- $spec
    echo "=== batched all modes nside=$1 B=$2 ==="
    python bench/bench_batched_modes.py --nside $1 --batch $2 \
        --ducc-threads 64 \
        --output results/batchmodes_n${1}_B${2}_v04node.json
done
echo "ALL DONE"
