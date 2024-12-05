import os
import pickle
import re

import hydra
from omegaconf import DictConfig


def processing_phrases(phrase: str) -> str:
    if isinstance(phrase, int):
        return str(phrase)  # deal with the int values
    return re.sub("[^A-Za-z0-9 ]", " ", phrase.lower()).strip()


def directory_exists(path: str) -> None:
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def construct_kgc_dataset(triplets: dict, save_path: str) -> None:
    extracted_triples = [[h, r, t] for (h, t), r in triplets.items()]
    with open(save_path, "w") as f:
        for triple in extracted_triples:
            f.write(",".join(triple) + "\n")


@hydra.main(
    config_path="config", config_name="stage1_kg_construction", version_base=None
)
def main(cfg: DictConfig) -> None:
    dataset = cfg.dataset.data_name
    extraction_type = cfg.task.openie_cfg.type

    graph_type = cfg.task.create_graph.graph_type
    phrase_type = cfg.task.create_graph.phrase_type
    version = cfg.task.create_graph.version
    retriever_name = cfg.task.create_graph.smodel
    processed_retriever_name = retriever_name.replace("/", "_").replace(".", "")

    file_path = f"data/{dataset}/tmp_merge/{dataset}_{graph_type}_graph_relation_dict_{phrase_type}_{extraction_type}_{processed_retriever_name}.{version}.subset.p"

    triplets = pickle.load(open(file_path, "rb"))

    save_path = f"data/{dataset}/processed/stage1/kg_merge.txt"  # _merge

    directory_exists(save_path)
    construct_kgc_dataset(triplets, save_path)


if __name__ == "__main__":
    main()
