#!/bin/bash
#SBATCH --job-name=deepseek_aoc_generation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --mem=15G
#SBATCH --partition=gpushort

echo "Starting job on $(hostname)"
echo "Time: $(date)"

set -e

module load Python/3.13.5-GCCcore-14.3.0
module load uv

cd ~/scriptie/habrok

source .env
uv run -m generation.main

echo "Finished at $(date)"
