#!/bin/bash
#SBATCH --job-name=aoc_generation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:a100:2
#SBATCH --time=06:00:00
#SBATCH --mem=15G
#SBATCH --partition=gpumedium

#!/bin/bash
echo "Starting job on $(hostname)"
echo "Time: $(date)"

set -e

module load Python/3.13.5-GCCcore-14.3.0
module load uv

cd ~/scriptie/habrok

source .env
uv run python3 -m data_generation.run_local \
    --years 2021 2024 \
    --days 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
    --batch-size 8 \
    --max-new-tokens 8192 \
    "$@"

echo "Finished at $(date)"
