# Fisher-Weighted Riemannian Model Merging for Orthogonal Finetuning

Code for the paper *Fisher-Weighted Riemannian Model Merging for Orthogonal Finetuning*.
Experiments use *Llama 3.1 8B* and *Qwen 2.5 3B* on twelve generative tasks, with an
*SDXL* study in the appendix.

Released finetunes: [https://anonymous-hf.com/a/rzhmxiutrwjg/](https://huggingface.co/eduardo-terres2/OFT-Llama-Qwen-Finetunes)

Run every command from the repository root. The SLURM scripts target `gpu_a100` /
`gpu_h100` partitions; adjust their `#SBATCH` headers for your cluster.

## 1. Install

The scripts activate these environments by name.

```bash
# merge: finetuning, FIM, merging, analysis, plots
conda create -n merge python=3.11 -y && conda activate merge
pip install -e .

# lm-eval: LLM evaluation
conda create -n lm-eval python=3.11 -y && conda activate lm-eval
pip install -e "OrthoMerge/eval/lm-evaluation-harness[hf,math,sentencepiece,unitxt]" \
    -r OrthoMerge/requirements.txt "pyarrow<20" "setuptools<81" "tensorflow==2.18.1" \
    bert-score nltk absl-py "git+https://github.com/google-research/bleurt.git"

# orthofuse_env: SDXL appendix only
conda env create -f OrthoFuse/environment.yml
```

Plots need a TeX install with `type1cm`, `cm-super` and `dvipng`. Without one, set
`text.usetex = False` in `src/analysis/plot_utils.py`.

## 2. Paths

```bash
export REPO_ROOT=$(pwd)                       # sbatch runs a copy of each script, so it cannot locate the repo itself
export MODELS_ROOT=/your/path/models          # base models + OFT adapters
export FISHERS_ROOT=/your/path/fishers        # LLM diagonal FIMs
export MERGED_MODELS_ROOT=/your/path/merged   # merged adapters

# SDXL appendix only
export SDXL_ROOT=/your/path/SDXL
export SDXL_FISHER_ROOT=/your/path/sdxl_fim
export HF_HOME=/your/path/hf-cache
export DIFFUSION_OUTPUT_DIR=/your/path/diffusion-outputs   # optional
```

Defaults are in `src/paths.py` and `src/diffusion/dataset.py`. Outputs go to `outputs/`,
datasets are cached in `data/hf_cache/`.

SLURM does not create the `#SBATCH --output` directories, so create them once:

```bash
mkdir -p outputs/oft_neighborhood_certificate outputs/evaluation_subsets/slurm \
    outputs/evaluation/{diagonal_fisher,standard_rescaled,baselines}/slurm \
    outputs/{fim,finetunes,eval_loss}/slurm outputs/analysis/subsets outputs/diffusion/slurms
```

Expected layout:

```
$MODELS_ROOT/Llama-3.1-8B/
$MODELS_ROOT/Llama-3.1-8B_OFT_adapters/llama3-1_8b_finetune_<task>/
$MODELS_ROOT/Qwen-2.5-3B/
$MODELS_ROOT/Qwen-2.5-3B_OFT_adapters/qwen2.5_3b_finetune_<task>/
$FISHERS_ROOT/<llama3.1|qwen2.5>/<task>/<adapter_tag>_<finetuned|pretrained>.safetensors
```

`<task>` is one of: `coqa`, `nq_open`, `triviaqa`, `meddialog_qsumm`, `wmt16-en-de`,
`wikitext`, `cnn_dailymail`, `xsum`, `babi`, `squadv2`, `mbpp`, `numinamath`.

## 3. Models and data

| What | How |
|---|---|
| Base LLMs | `python utils/download_weights.py` (Llama is gated: `huggingface-cli login` first) |
| OFT finetunes | Download from https://anonymous-hf.com/a/rzhmxiutrwjg/ into `$MODELS_ROOT` |
| LLM diagonal FIMs | Recompute (section 4) |
| Task datasets | Downloaded automatically on first use |
| SDXL backbone | Downloaded into `$HF_HOME`; run once with `HF_HUB_OFFLINE=0` |
| SDXL GSOFT adapters + images | From the OrthoFuse release (see `OrthoFuse/README.md`) |

SDXL assets go under `$SDXL_ROOT`:
`concepts/adapters/pytorch_lora_weights_<concept>.safetensors`, `concepts/datasets/<concept>/`,
`styles/adapters/pytorch_lora_weights_<style>.safetensors`, `styles/datasets/<image>`.

## 4. Finetuning and FIMs

Skip finetuning if you use the released adapters.

```bash
conda activate merge
sbatch scripts/finetune_parallel.sh   # array 0-11; set MODEL_FAMILY inside
sbatch scripts/compute_fim.sh         # array 0-11; set MODEL_NAME inside
```

Hyperparameters are at the top of `src/finetune/finetune.py`. For both FIM anchors pass
`--model-states finetuned pretrained` to `compute_fim.sh`.

SDXL adapters are trained with `OrthoFuse/train.sh` and `OrthoFuse/train_style.sh`; their
FIMs with `sbatch scripts/diffusion/compute_fim.sh` (array 0-17).

## 5. Main paper

### Figure 2: Convex normal neighborhood

```bash
sbatch scripts/analysis/oft_neighborhood_certificate.sh
```

Needs only the adapters. Output: `outputs/oft_neighborhood_certificate/combined_max_block_distance_mds.pdf`.

### Table 1: Task addition

```bash
sbatch scripts/eval/pipe_finetunes.sh     # set MODEL_NAME inside
sbatch scripts/eval/pipe_fishermerge.sh   # set MODEL_NAME + FISHER_SOURCE inside
sbatch scripts/eval/pipe_orthomerge.sh    # MERGE_MODE=standard for the Lie sum
sbatch scripts/eval/pipe_baselines.sh     # TIES, TSVM, Wudi (array 0-5)

bash scripts/llm_table_results.sh --format latex
```

Run `pipe_fishermerge.sh` with `FISHER_SOURCE=finetuned` and `FISHER_SOURCE=pretrained`.
Results go to `outputs/evaluation/<method>/<family>/<task>/`. The `METHODS` array in
`llm_table_results.sh` must match the runs you did.

### Figure 3: Number of merged tasks

```bash
python -m src.subsets.subsets_merge --phase merge \
    --model-family llama3.1 --models-dir outputs/models/subsets/llama3.1
sbatch scripts/subsets/subsets_pipe.sh
sbatch scripts/subsets/subsets_plots.sh
```

Output: `outputs/subsets/combined/subset_combined_only_included.pdf`.

## 6. Appendix

### Tables 2 and 3: OFT adapter block structure

```bash
sbatch scripts/oft_info.sh                     # Table 2
sbatch scripts/diffusion/oft_adapter_info.sh   # Table 3
```

### Table 4: FIM sizes

```bash
conda activate merge
python -m src.info
```

### Figure 7: Per-dataset subset metrics

Produced by `sbatch scripts/subsets/subsets_plots.sh`. Output:
`outputs/subsets/<family>/subset_dataset_metrics_{llama,qwen}.pdf`.

### Figures 8 and 9: Geodesic mode connectivity

```bash
sbatch scripts/analysis/loss_interpolation.sh
```

Output: `outputs/interpolation/<family>/imgs/loss_interpolation_{qwen,llama}_0_2.pdf`.

### Figure 10: FIM differences and normalization

```bash
sbatch scripts/diffusion/analysis/normalize_fim_style_average.sh
```

Needs the SDXL FIMs. Output: `outputs/diffusion/analysis/fim/`.

### Figures 11 and 14: Interpolation Pareto frontier and examples

```bash
sbatch scripts/diffusion/geodesic_interpolation.sh              # array 0-11 over styles
sbatch scripts/diffusion/plots/plot_geodesic_interpolation.sh
```

Run the generation once per entry of the `METHODS` array in the first script. Figure 14
uses montages from `samples_interpolation_geodesic/`.
