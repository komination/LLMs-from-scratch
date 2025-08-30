import json
import os
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download


repo_id = f"Qwen/Qwen3-0.6B"

local_dir = Path(repo_id).parts[-1]

weights_file = hf_hub_download(
    repo_id=repo_id,
    filename="model.safetensors",
    local_dir=local_dir,
)