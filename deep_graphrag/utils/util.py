import json
import os

import torch
from hydra.utils import get_class
from omegaconf import DictConfig

from deep_graphrag.datasets import KGDataset


def load_model_from_pretrained(path: str) -> tuple[torch.nn.Module, dict]:
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    model = get_class(config["architectures"])(**config["model_config"])
    state = torch.load(os.path.join(path, "model.pth"), map_location="cpu")
    model.load_state_dict(state["model"])
    return model, config


def build_pretrain_dataset(cfg: DictConfig) -> list[KGDataset]:
    """
    Return the joint KG datasets
    """
    dataset_cfg = {k: v for k, v in cfg.datasets.items() if k != "data_names"}
    data_name_list = cfg.datasets.data_names
    dataset_list = []
    for data_name in data_name_list:
        kg_data = KGDataset(**dataset_cfg, data_name=data_name)[0]
        dataset_list.append(kg_data)
    return dataset_list
