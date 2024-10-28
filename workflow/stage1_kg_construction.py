import json
import os
import re
from glob import glob

import hydra
from omegaconf import DictConfig


def processing_phrases(phrase: str) -> str:
    return re.sub("[^A-Za-z0-9 ]", " ", phrase.lower()).strip()


def directory_exists(path: str) -> None:
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def construct_kgc_dataset(extracted_triples: dict, save_path: str) -> None:
    with open(save_path, "w") as f:
        for sample in extracted_triples:
            triples = sample["extracted_triples"]
            for lst in triples:  # TODO: judge whether len(lst) == 3
                try:
                    lst = [processing_phrases(t) for t in lst]
                    if len(lst) == 3 and "" not in lst:
                        f.write(",".join(lst) + "\n")
                except Exception as e:
                    print(e)
                    print(lst)


@hydra.main(
    config_path="config", config_name="stage1_kg_construction", version_base=None
)
def main(cfg: DictConfig) -> None:
    dataset = cfg.dataset.data_name
    extraction_type = cfg.task.openie_cfg.type
    model_name = cfg.task.openie_cfg.llm

    # corpus ner results
    possible_files = glob(
        f"data/{dataset}/tmp/openie_{dataset}_results_{extraction_type}_{model_name}_*.json"
    )
    extracted_file = json.load(open(possible_files[0]))
    extracted_triples = extracted_file["docs"]
    save_path = f"data/{dataset}/processed/stage1/kg.txt"

    directory_exists(save_path)
    construct_kgc_dataset(extracted_triples, save_path)


if __name__ == "__main__":
    main()
