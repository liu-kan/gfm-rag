import json
import logging
import os

import torch
from sentence_transformers import SentenceTransformer
from torch.utils import data as torch_data
from torch_geometric.data import InMemoryDataset

from deep_graphrag.datasets.kg_dataset import KGDataset
from deep_graphrag.utils import get_rank, is_main_process, synchronize
from deep_graphrag.utils.qa_utils import entities_to_mask

logger = logging.getLogger(__name__)


class QADataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        data_name: str,
        text_emb_model_name: str,
    ):
        self.name = data_name
        self.text_emb_model_name = text_emb_model_name
        self.kg = KGDataset(root, data_name, text_emb_model_name)[0]
        super().__init__(root, None, None)
        self.data = torch.load(self.processed_paths[0], weights_only=False)

    def __repr__(self) -> str:
        return f"{self.name}()"

    @property
    def raw_file_names(self) -> list:
        return ["train.json", "test.json"]

    @property
    def raw_dir(self) -> str:
        return os.path.join(str(self.root), str(self.name), "processed", "stage1")

    @property
    def processed_dir(self) -> str:
        return os.path.join(str(self.root), str(self.name), "processed", "stage2")

    @property
    def processed_file_names(self) -> str:
        return "qa_data.pt"

    def _process(self) -> None:
        if is_main_process():
            logger.info("Processing QA dataset at rank %d", get_rank())
            super()._process()
        else:
            logger.info(
                f"Rank [{get_rank()}]: Waiting for main process to finish processing QA dataset"
            )
        synchronize()

    def process(self) -> None:
        with open(os.path.join(self.processed_dir, "ent2id.json")) as fin:
            self.ent2id = json.load(fin)
        with open(os.path.join(self.processed_dir, "rel2id.json")) as fin:
            self.rel2id = json.load(fin)

        num_nodes = self.kg.num_nodes
        questions = []
        question_entities_masks = []  # Convert question entities to mask with number of nodes
        supporting_entities_masks = []
        num_samples = []

        for path in self.raw_paths:
            with open(path) as fin:
                data = json.load(fin)
                num_samples.append(len(data))
                for item in data:
                    question = item["question"]
                    questions.append(question)

                    question_entities = [
                        self.ent2id[x]
                        for x in item["question_entities"]
                        if x in self.ent2id
                    ]

                    question_entities_masks.append(
                        entities_to_mask(question_entities, num_nodes)
                    )

                    supporting_entities = [
                        self.ent2id[x]
                        for x in item["supporting_entities"]
                        if x in self.ent2id
                    ]

                    supporting_entities_masks.append(
                        entities_to_mask(supporting_entities, num_nodes)
                    )

        # Generate question embeddings
        logger.info("Generating question embeddings")
        text_emb_model = SentenceTransformer(self.text_emb_model_name)
        question_embeddings = text_emb_model.encode(
            questions,
            device="cuda" if torch.cuda.is_available() else "cpu",
            show_progress_bar=True,
            convert_to_tensor=True,
        ).cpu()
        question_entities_masks = torch.stack(question_entities_masks)
        supporting_entities_masks = torch.stack(supporting_entities_masks)

        dataset = torch_data.TensorDataset(
            question_embeddings, question_entities_masks, supporting_entities_masks
        )
        offset = 0
        splits = []
        for num_sample in num_samples:
            split = torch_data.Subset(dataset, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        torch.save(splits, self.processed_paths[0])
