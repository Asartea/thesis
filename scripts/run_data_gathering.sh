#!/bin/bash
echo "Time: $(date)"

set -e

source .env

uv run python3 -m data_gathering.main

echo "Finished at $(date)"
