import json
import os

import torch
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf


def save_model_to_pretrained(
    model: torch.nn.Module, cfg: DictConfig, path: str
) -> None:
    os.makedirs(path, exist_ok=True)
    config = {
        "architectures": cfg.model._target_,
        "text_emb_model": cfg.datasets.cfgs.text_emb_model_name,
        "model_config": {
            "rel_emb_dim": model.rel_emb_dim,
            "entity_model_cfg": OmegaConf.to_container(cfg.model.entity_model_cfg),
        },
    }

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    torch.save({"model": model.state_dict()}, os.path.join(path, "model.pth"))


def load_model_from_pretrained(path: str) -> tuple[torch.nn.Module, dict]:
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    model = get_class(config["architectures"])(**config["model_config"])
    state = torch.load(os.path.join(path, "model.pth"), map_location="cpu")
    model.load_state_dict(state["model"])
    return model, config


def get_multi_dataset(cfg: DictConfig) -> dict:
    """
    Return the joint KG datasets
    """
    data_name_list = cfg.datasets.data_names
    dataset_cls = get_class(cfg.datasets._target_)
    dataset_list = {}
    for data_name in data_name_list:
        kg_data = dataset_cls(**cfg.datasets.cfgs, data_name=data_name)
        dataset_list[data_name] = kg_data
    return dataset_list
