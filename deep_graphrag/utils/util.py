import json
import os

import torch
from hydra.utils import get_class


def load_model_from_pretrained(path: str) -> tuple[torch.nn.Module, dict]:
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    model = get_class(config["architectures"])(**config["model_config"])
    state = torch.load(os.path.join(path, "model.pth"), map_location="cpu")
    model.load_state_dict(state["model"])
    return model, config
