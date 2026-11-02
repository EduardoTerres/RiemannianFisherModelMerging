from huggingface_hub import snapshot_download
import os

MODELS_DIR = os.environ.get("MODELS_ROOT", "/scratch-shared/USERNAME/models")

BASE_MODELS = {
    "meta-llama/Llama-3.1-8B": f"{MODELS_DIR}/Llama-3.1-8B",
    "Qwen/Qwen2.5-3B": f"{MODELS_DIR}/Qwen-2.5-3B",
}

os.makedirs(MODELS_DIR, exist_ok=True)

for repo_id, save_folder in BASE_MODELS.items():
    print(f"Downloading {repo_id} -> {save_folder}")
    snapshot_download(
        repo_id=repo_id, local_dir=save_folder, local_dir_use_symlinks=False
    )
