#!/bin/bash
echo "Starting job on $(hostname)"
echo "Time: $(date)"

set -e

module load Python/3.13.5-GCCcore-14.3.0
module load uv

cd ~/scriptie/habrok

source .env
uv run -m data_generation.main

echo "Finished at $(date)"
