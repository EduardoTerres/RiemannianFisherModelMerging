#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --output=outputs/fim/slurm/fim_%A_%a.out

set -e

# Repo root, derived from this script's location (override by exporting REPO_ROOT).
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET_CACHE_DIR="${REPO_ROOT}/data/hf_cache"
mkdir -p "${DATASET_CACHE_DIR}"

MODEL_NAME="llama3.1"  # "llama3.1" or "qwen2.5"
SAVE_NAME=""  # optional: overrides the adapter tag used in the saved FIM filenames

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
EXTRA_ARGS=()
if [[ -n "${SAVE_NAME}" ]]; then
    EXTRA_ARGS+=(--save-name "${SAVE_NAME}")
fi

python "${REPO_ROOT}/src/compute_fim.py" \
    --model-family "${MODEL_NAME}" \
    --task-index "${TASK_INDEX}" \
    --num-samples 4096 \
    --batch-size 1 \
    --dataset-cache-dir "${DATASET_CACHE_DIR}" \
    --force-compute \
    --model-states finetuned \
    "${EXTRA_ARGS[@]}"
