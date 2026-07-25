# Fisher-Weighted Riemannian Model Merging for Orthogonal Fine-Tuning

Research code for fine-tuning, merging, and evaluating OFT adapters for causal
language models. The main experiment path is:

1. Fine-tune one OFT adapter per task.
2. Optionally compute Fisher matrices for those adapters.
3. Merge adapters with standard or Fisher-weighted Riemannian methods.
4. Evaluate merged adapters with target-token loss or lm-eval.

## Layout

- `src/merging.py`: core OFT/Riemannian merging methods.
- `src/geometry.py`: manifold operations used by merging.
- `src/dataset/dataset.py`: task list, prompt formatting, and dataloaders.
- `src/finetune/finetune.py`: fine-tune task adapters.
- `src/compute_fim.py`: compute Fisher matrices.
- `src/scripts/perform_merging.py`: create merged adapters/models.
- `src/loss/compute_eval_loss.py`: evaluate target-token loss.
- `src/analysis/` and `src/plots/`: analysis and figures.
- `scripts/`: SLURM wrappers for cluster runs.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Model, adapter, and Fisher paths are configured in `src/paths.py`. By default the
repo expects base models under `data/models/` and task adapters in folders named
like:

```text
data/models/Llama-3.1-8B_OFT_dataset_adapters
data/models/Qwen-2.5-3B_OFT_dataset_adapters
```

Update `FISHERS_DIR` in `src/paths.py` before running Fisher-based workflows.

## Common Commands

List tasks:

```bash
python -m src.finetune.finetune --list-tasks
```

Fine-tune one adapter:

```bash
python -m src.finetune.finetune --model-family llama3.1 --task coqa
```

Compute Fisher matrices:

```bash
python -m src.compute_fim --model-family llama3.1 --task-index 0
```

Merge adapters:

```bash
python -m src.scripts.perform_merging \
  --model_family llama3.1 \
  --merge_method gradients \
  --merge_mode standard_rescaled \
  --output_dir outputs/merged/llama3.1_standard_rescaled
```

Evaluate loss:

```bash
python -m src.loss.compute_eval_loss \
  --model llama3.1 \
  --model-source adapter \
  --peft-model outputs/merged/llama3.1_standard_rescaled/merged_adapter
```

Run tests:

```bash
pytest
```

## Notes

Most heavy workflows assume CUDA, Hugging Face access, and enough GPU memory for
the selected model family. The SLURM scripts in `scripts/` are useful templates,
but contain cluster-specific partitions and paths that may need editing.

Generated models, adapters, Fisher tensors, caches, and evaluation outputs should
stay out of git.
