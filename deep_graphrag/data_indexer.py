import json
import logging
import os

from omegaconf import DictConfig

from .kg_construction import BaseKGConstructor, BaseQAConstructor
from .kg_construction.utils import KG_DELIMITER

logger = logging.getLogger(__name__)


class DataIndexer:
    DELIMITER = KG_DELIMITER

    def __init__(
        self, kg_constructor: BaseKGConstructor, qa_constructor: BaseQAConstructor
    ) -> None:
        self.kg_constructor = kg_constructor
        self.qa_constructor = qa_constructor

    def index_data(self, dataset_cfg: DictConfig) -> None:
        root = dataset_cfg.root
        data_name = dataset_cfg.data_name
        raw_data_dir = os.path.join(root, data_name, "raw")
        prosessed_data_dir = os.path.join(root, data_name, "processed", "stage1")

        if not os.path.exists(prosessed_data_dir):
            os.makedirs(prosessed_data_dir)

        # Create KG index for each dataset
        if not os.path.exists(os.path.join(prosessed_data_dir, "kg.txt")):
            logger.info("Stage1 KG construction")
            kg = self.kg_constructor.create_kg(root, data_name)
            with open(os.path.join(prosessed_data_dir, "kg.txt"), "w") as f:
                for triple in kg:
                    f.write(self.DELIMITER.join(triple) + "\n")
        if not os.path.exists(
            os.path.join(prosessed_data_dir, "document2entities.json")
        ):
            logger.info("Stage1 Get document2entities")
            doc2entities = self.kg_constructor.get_document2entities(root, data_name)
            with open(
                os.path.join(prosessed_data_dir, "document2entities.json"), "w"
            ) as f:
                json.dump(doc2entities, f, indent=4)

        # Try to prepare training and testing data from dataset
        if os.path.exists(
            os.path.join(raw_data_dir, "train.json")
        ) and not os.path.exists(os.path.join(prosessed_data_dir, "train.json")):
            logger.info(f"Preparing {os.path.join(raw_data_dir, 'train.json')}")
            train_data = self.qa_constructor.prepare_data(root, data_name, "train.json")
            with open(os.path.join(prosessed_data_dir, "train.json"), "w") as f:
                json.dump(train_data, f, indent=4)

        if os.path.exists(
            os.path.join(raw_data_dir, "test.json")
        ) and not os.path.exists(os.path.join(prosessed_data_dir, "test.json")):
            logger.info(f"Preparing {os.path.join(raw_data_dir, 'test.json')}")
            test_data = self.qa_constructor.prepare_data(root, data_name, "test.json")
            with open(os.path.join(prosessed_data_dir, "test.json"), "w") as f:
                json.dump(test_data, f, indent=4)
