#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=pipe_fmerge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=outputs/evaluation/diagonal_fisher/slurm/pipe_fishermerge_%A.out

set -e

# MODEL_NAME="llama3.1"  # "llama3.1" or "qwen2.5"
MODEL_NAME="qwen2.5"
MODEL_NAME="llama3.1"

MERGE_METHOD="gradients"
MERGE_MODE="diagonal_fisher"
# FISHER_SOURCE selects which FIM to merge with: "finetuned", "pretrained",
# or "average" (average of the pretrained and finetuned FIM per dataset).
FISHER_SOURCE="finetuned"
# FIM_NORMALIZATION selects how the Fisher matrices are normalized before
# merging: "none", "trace", "frobenius", "layer-trace", "layer-frobenius", or "kl".
FIM_NORMALIZATION="none"
# Repo root, derived from this script's location (override by exporting REPO_ROOT).
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODELS_ROOT="${MODELS_ROOT:-/scratch-shared/USERNAME/models}"
# Where merged adapters are written and read back (override by exporting it).
MERGED_MODELS_ROOT="${MERGED_MODELS_ROOT:-/scratch-shared/USERNAME/merged_models}"


if [[ "${MODEL_NAME}" == "llama3.1" ]]; then
    BASE_MODEL_PATH="${MODELS_ROOT}/Llama-3.1-8B"
elif [[ "${MODEL_NAME}" == "qwen2.5" ]]; then
    BASE_MODEL_PATH="${MODELS_ROOT}/Qwen-2.5-3B"
else
    echo "Unknown model name: ${MODEL_NAME}"
    exit 1
fi

if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
    echo "Base model path does not exist: ${BASE_MODEL_PATH}"
    exit 1
fi

# Keep run naming backward-compatible: only tag the output paths with the
# normalization when it differs from the previous implicit default ("none").
RUN_TAG="${MERGE_MODE}_${FISHER_SOURCE}"
if [[ "${FIM_NORMALIZATION}" != "none" ]]; then
    RUN_TAG="${RUN_TAG}_fim_${FIM_NORMALIZATION}"
fi

MODEL_DIR="${MERGED_MODELS_ROOT}/${RUN_TAG}/${MODEL_NAME}/merged_model"
MERGED_ADAPTER_PATH="${MODEL_DIR}/merged_adapter"

EVAL_DIR="${REPO_ROOT}/outputs/evaluation/${RUN_TAG}/${MODEL_NAME}"
SLURM_DIR="${REPO_ROOT}/outputs/evaluation/${RUN_TAG}/slurm"

source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "${SLURM_DIR}"

conda activate merge

python "${REPO_ROOT}/src/scripts/perform_merging.py" \
    --model_family "${MODEL_NAME}" \
    --merge_method "${MERGE_METHOD}" \
    --merge_mode "${MERGE_MODE}" \
    --fisher_source "${FISHER_SOURCE}" \
    --fim_normalization "${FIM_NORMALIZATION}" \
    --output_dir "${MODEL_DIR}" \
    --save_merged_model \
    --lam 0.0 \
    --device gpu

sbatch \
    --output="${SLURM_DIR}/eval_model_%A_%a.out" \
    "${REPO_ROOT}/scripts/eval/eval.sh" \
    "${BASE_MODEL_PATH}" \
    "${EVAL_DIR}" \
    "${MERGED_ADAPTER_PATH}"
