import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from multiprocessing.dummy import Pool as ThreadPool

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from deep_graphrag.kg_construction.utils import KG_DELIMITER

from .entity_linking_model import BaseELModel
from .ner_model import BaseNERModel

logger = logging.getLogger(__name__)


class BaseQAConstructor(ABC):
    @abstractmethod
    def prepare_data(self, data_root: str, data_name: str, file: str) -> list[dict]:
        """
        Prepare QA data for training and evaluation

        Args:
            data_root (str): path to the dataset
            data_name (str): name of the dataset
            file (str): file name to process
        Returns:
            list[dict]: list of processed data
        """
        pass


class QAConstructor(BaseQAConstructor):
    DELIMITER = KG_DELIMITER

    def __init__(
        self,
        ner_model: BaseNERModel,
        el_model: BaseELModel,
        root: str = "tmp/qa_construnction",
        num_processes: int = 1,
        force: bool = False,
    ) -> None:
        self.ner_model = ner_model
        self.el_model = el_model
        self.root = root
        self.num_processes = num_processes
        self.data_name = None
        self.force = force

    @property
    def tmp_dir(self) -> str:
        assert (
            self.data_name is not None
        )  # data_name should be set before calling this property
        tmp_dir = os.path.join(self.root, self.data_name)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        return tmp_dir

    @staticmethod
    def from_config(cfg: DictConfig) -> "QAConstructor":
        # create a fingerprint of config for tmp directory
        config = OmegaConf.to_container(cfg, resolve=True)
        if "force" in config:
            del config["force"]
        fingerprint = hashlib.md5(json.dumps(config).encode()).hexdigest()

        base_tmp_dir = os.path.join(cfg.root, fingerprint)
        if not os.path.exists(base_tmp_dir):
            os.makedirs(base_tmp_dir)
            json.dump(
                config,
                open(os.path.join(base_tmp_dir, "config.json"), "w"),
                indent=4,
            )
        return QAConstructor(
            root=base_tmp_dir,
            ner_model=instantiate(cfg.ner_model),
            el_model=instantiate(cfg.el_model),
            num_processes=cfg.num_processes,
            force=cfg.force,
        )

    def prepare_data(self, data_root: str, data_name: str, file: str) -> list[dict]:
        # Get dataset information
        self.data_name = data_name  # type: ignore
        raw_path = os.path.join(data_root, data_name, "raw", file)
        processed_path = os.path.join(data_root, data_name, "processed", "stage1")

        if self.force:
            # Clear cache in tmp directory
            for tmp_file in os.listdir(self.tmp_dir):
                os.remove(os.path.join(self.tmp_dir, tmp_file))

        if not os.path.exists(os.path.join(processed_path, "kg.txt")):
            raise FileNotFoundError(
                "KG file not found. Please run KG construction first"
            )

        # Read KG
        entities = set()
        with open(os.path.join(processed_path, "kg.txt")) as f:
            for line in f:
                try:
                    u, _, v = line.strip().split(self.DELIMITER)
                except Exception as e:
                    logger.error(f"Error in line: {line}, {e}, Skipping")
                    continue
                entities.add(u)
                entities.add(v)
        # Read document2entities
        with open(os.path.join(processed_path, "document2entities.json")) as f:
            doc2entities = json.load(f)

        # Load data
        with open(raw_path) as f:
            data = json.load(f)

        ner_results = {}
        # Try to read ner results
        if os.path.exists(os.path.join(self.tmp_dir, "ner_results.jsonl")):
            with open(os.path.join(self.tmp_dir, "ner_results.jsonl")) as f:
                ner_logs = [json.loads(line) for line in f]
                ner_results = {log["id"]: log for log in ner_logs}

        unprocessed_data = [
            sample for sample in data if sample["id"] not in ner_results
        ]

        def _ner_process(sample: dict) -> dict:
            id = sample["id"]
            question = sample["question"]
            ner_ents = self.ner_model(question)
            return {
                "id": id,
                "question": question,
                "ner_ents": ner_ents,
            }

        # NER
        with ThreadPool(self.num_processes) as pool:
            with open(os.path.join(self.tmp_dir, "ner_results.jsonl"), "a") as f:
                for res in tqdm(
                    pool.imap(_ner_process, unprocessed_data),
                    total=len(unprocessed_data),
                ):
                    ner_results[res["id"]] = res
                    f.write(json.dumps(res) + "\n")

        # EL
        self.el_model.index(list(entities))

        ner_entities = []
        for res in ner_results.values():
            ner_entities.extend(res["ner_ents"])

        el_results = self.el_model(ner_entities, topk=1)

        # Prepare final data
        final_data = []
        for sample in data:
            id = sample["id"]
            ner_ents = ner_results[id]["ner_ents"]
            question_entities = []
            for ent in ner_ents:
                question_entities.append(el_results[ent][0]["entity"])

            supporting_facts = sample.get("supporting_facts", [])
            supporting_entities = []
            for item in list(set(supporting_facts)):
                supporting_entities.extend(doc2entities.get(item, []))

            final_data.append(
                {
                    **sample,
                    "question_entities": question_entities,
                    "supporting_entities": supporting_entities,
                }
            )

        return final_data
