import os
from dataclasses import dataclass
from pathlib import Path

from src.dataset.dataset import DATASET_TRAIN

RELATIVE_ROOTDIR = Path("../..")
ROOTDIR = (Path(__file__) / RELATIVE_ROOTDIR).resolve()

WANDB_PROJECT = "fisher-orthomerge"

@dataclass(frozen=True)
class ModelFamily:
    name: str
    base_model_path: str
    adapter_paths: list
    fisher_finetuned_paths: list
    fisher_pretrained_paths: list
    default_fisher: str = "finetuned"

    @property
    def fisher_paths(self):
        if self.default_fisher == "pretrained":
            return self.fisher_pretrained_paths
        if self.default_fisher == "average":
            return self._average_fisher_paths()
        return self.fisher_finetuned_paths

    def _average_fisher_paths(self) -> list:
        """Average the pretrained/finetuned FIM of each dataset and cache the result to disk."""
        from safetensors.torch import load_file, save_file

        out_paths = []
        for ft_path, pt_path in zip(self.fisher_finetuned_paths, self.fisher_pretrained_paths):
            out_path = ft_path.replace("_finetuned.safetensors", "_average.safetensors")
            if not Path(out_path).exists():
                ft, pt = load_file(ft_path), load_file(pt_path)
                save_file({k: (ft[k] + pt[k]) / 2 for k in ft}, out_path)
            out_paths.append(out_path)
        return out_paths

    def __str__(self):
        return self.name

# Base models, OFT adapters and Fisher matrices. Override with the matching
# environment variable instead of editing this file.
MODELS_DIR = Path(os.environ.get("MODELS_ROOT", "/scratch-shared/USERNAME/models"))
FISHERS_DIR = Path(os.environ.get("FISHERS_ROOT", "/scratch-shared/USERNAME/fishers"))


def fisher_path(model_family: str, task: str, adapter_tag: str, model_state: str) -> str:
    return f"{FISHERS_DIR}/{model_family}/{task}/{adapter_tag}_{model_state}.safetensors"


def with_default_fisher(families: dict[str, ModelFamily], default_fisher: str) -> dict[str, ModelFamily]:
    return {
        name: ModelFamily(
            name=family.name,
            base_model_path=family.base_model_path,
            adapter_paths=family.adapter_paths,
            fisher_finetuned_paths=family.fisher_finetuned_paths,
            fisher_pretrained_paths=family.fisher_pretrained_paths,
            default_fisher=default_fisher,
        )
        for name, family in families.items()
    }

# LLAMA ADAPTERS, FISHERS AND TASKS
LLAMA_MODEL_NAME = "Llama-3.1-8B"
LLAMA_BASE_MODEL_PATH = f"{MODELS_DIR}/Llama-3.1-8B"
LLAMA_ADAPTERS_FOLDER = f"{MODELS_DIR}/Llama-3.1-8B_OFT_adapters"
LLAMA_TASKS = [tag for tag, *_ in DATASET_TRAIN]
LLAMA_ADAPTER_PATHS = [
    f"{LLAMA_ADAPTERS_FOLDER}/llama3-1_8b_finetune_{task}"
    for task in LLAMA_TASKS
]
LLAMA_FISHER_PATHS = [
    fisher_path(
        "llama3.1", task, f"llama3-1_8b_finetune_{task}", "finetuned"
    )
    for task in LLAMA_TASKS
]
LLAMA_PRETRAINED_FISHER_PATHS = [
    fisher_path(
        "llama3.1", task, f"llama3-1_8b_finetune_{task}", "pretrained"
    )
    for task in LLAMA_TASKS
]

# QWEN ADAPTERS, FISHERS AND TASKS
QWEN_MODEL_NAME = "Qwen-2.5-3B"
QWEN_BASE_MODEL_PATH = f"{MODELS_DIR}/Qwen-2.5-3B"
QWEN_ADAPTERS_FOLDER = f"{MODELS_DIR}/Qwen-2.5-3B_OFT_adapters"
QWEN_TASKS = [tag for tag, *_ in DATASET_TRAIN]
QWEN_ADAPTER_PATHS = [
    f"{QWEN_ADAPTERS_FOLDER}/qwen2.5_3b_finetune_{task}"
    for task in QWEN_TASKS
]
QWEN_FISHER_PATHS = [
    fisher_path(
        "qwen2.5", task, f"qwen2.5_3b_finetune_{task}", "finetuned"
    )
    for task in QWEN_TASKS
]
QWEN_PRETRAINED_FISHER_PATHS = [
    fisher_path(
        "qwen2.5", task, f"qwen2.5_3b_finetune_{task}", "pretrained"
    )
    for task in QWEN_TASKS
]

# MODEL FAMILIES for importing
MODEL_FAMILIES_BASE = {
    "llama3.1": ModelFamily(
        name="llama3.1",
        base_model_path=LLAMA_BASE_MODEL_PATH,
        adapter_paths=LLAMA_ADAPTER_PATHS,
        fisher_finetuned_paths=LLAMA_FISHER_PATHS,
        fisher_pretrained_paths=LLAMA_PRETRAINED_FISHER_PATHS,
    ),
    "qwen2.5": ModelFamily(
        name="qwen2.5",
        base_model_path=QWEN_BASE_MODEL_PATH,
        adapter_paths=QWEN_ADAPTER_PATHS,
        fisher_finetuned_paths=QWEN_FISHER_PATHS,
        fisher_pretrained_paths=QWEN_PRETRAINED_FISHER_PATHS,
    ),
}

MODEL_FAMILIES_FISHER_FINETUNES = with_default_fisher(MODEL_FAMILIES_BASE, "finetuned")
MODEL_FAMILIES_FISHER_PRETRAINED = with_default_fisher(MODEL_FAMILIES_BASE, "pretrained")
MODEL_FAMILIES_FISHER_AVERAGE = with_default_fisher(MODEL_FAMILIES_BASE, "average")
MODEL_FAMILIES = MODEL_FAMILIES_FISHER_FINETUNES
