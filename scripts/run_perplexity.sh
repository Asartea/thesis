#!/bin/bash
#SBATCH --job-name=aoc_perplexity
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:a100:2
#SBATCH --time=04:00:00
#SBATCH --mem=15G
#SBATCH --partition=gpushort

#!/bin/bash
echo "Starting job on $(hostname)"
echo "Time: $(date)"

set -e

module load Python/3.13.5-GCCcore-14.3.0
module load uv

cd ~/scriptie/habrok

source .env
uv run python3 -m detection.perplexity.fast_detect_gpt \
    "$@"


echo "Finished at $(date)"
