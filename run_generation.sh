#!/bin/bash
#SBATCH --job-name=qwen_aoc_generation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --partition=regular

echo "Starting job on $(hostname)"
echo "Time: $(date)"

set -e

module load EESSI/2025.06
module load Python/3.13.5-GCCcore-14.3.0
module load uv

cd ~/scriptie/habrok

source .env
pwd
source .venv/bin/activate
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
#uv run generation/main.py

echo "Finished at $(date)"
