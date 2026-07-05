#!/bin/bash
#SBATCH -A m4895
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 128
#SBATCH -J almond_bench
#SBATCH -o logs/%x_%j.out

# Benchmark Almond GPU synthesis vs ducc0 CPU on a dedicated A100 node.
# The node has 64 physical cores (128 hyperthreads); ducc gets up to 64.

set -e
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate simaster

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/.."
mkdir -p bench/logs results

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "cores: $(nproc)"

for nside in 128 256 512 1024 2048; do
    echo "=== nside=$nside ==="
    python bench/bench_synthesis.py --nside $nside \
        --ducc-threads 1 16 32 64 \
        --output results/synth_n${nside}_debugnode.json
done

# high-throughput batched comparison: Almond vs ducc ntrans (64 threads)
for spec in "256 128" "512 64" "1024 64" "2048 16"; do
    set -- $spec
    echo "=== batched nside=$1 B=$2 ==="
    python bench/bench_batched.py --nside $1 --batch $2 --ducc-threads 64 \
        --output results/batch_n${1}_B${2}_debugnode.json
done

# v0.2: adjoint + spin-2 vs ducc 64t
for nside in 512 1024 2048; do
    echo "=== v02 nside=$nside ==="
    python bench/bench_v02.py --nside $nside \
        --output results/v02_n${nside}_debugnode.json
done
echo "ALL DONE"
