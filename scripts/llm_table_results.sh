#!/bin/bash

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate merge

# Entries should be METHOD or METHOD=DISPLAY_LABEL, where METHOD is the output folder name.
METHODS=(
  "finetunes=Finetuned"
  "diagonal_fisher_pretrained=Diagonal Fisher at pretrained"
  "diagonal_fisher_finetuned=Diagonal Fisher at finetuned"
  "standard=Lie sum (alpha=1)"
  "standard_rescaled=OrthoMerge"
  "orthomerge_c_ties=TIES"
  "orthomerge_c_tsvm=TSVM"
  "wudi=Wudi merging"
)

python -m src.analysis.performance_table \
  --decimals 2 \
  --methods "${METHODS[@]}" \
  "$@"
