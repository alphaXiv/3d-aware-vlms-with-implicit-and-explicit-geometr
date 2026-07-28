#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "RUN_COMMAND=bash scripts/run_reproduction.sh"
echo "QUALIFYING_RECOVERY_CUTOFF_UTC=2026-07-28T01:51:38.616Z"
echo "BACKEND=kubernetes"
echo "EXPECTED_GPU_MODEL=NVIDIA RTX PRO 6000 Blackwell"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

python -m pip install --quiet --no-cache-dir -r requirements-repro.txt
python -m src.reproduce --config configs/variant.json
