# mypy: ignore-errors
import json
import os

import torch
from sentence_transformers import SentenceTransformer
from torch.utils import data as torch_data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.separate import separate

from .qa_utils import entities_to_mask


class QADataset(InMemoryDataset):
    def __init__(
        self,
        root,
        emb_model_name,
        graph_path,
        transform=None,
        pre_transform=None,
        **kwargs,
    ):
        self.emb_model_name = emb_model_name
        self.graph_path = graph_path
        # Load Graph info
        data = torch.load(os.path.join(self.graph_path, "data.pt"))
        self.graph = separate(
            cls=data[0].__class__,
            batch=data[0],
            idx=0,
            slice_dict=data[1],
            decrement=False,
        )  # Use the training graph
        super().__init__(root, transform, pre_transform)

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        with open(os.path.join(self.graph_path, "ent2id.json")) as fin:
            self.ent2id = json.load(fin)
        with open(os.path.join(self.graph_path, "rel2id.json")) as fin:
            self.rel2id = json.load(fin)

        train_path = os.path.join(self.raw_dir, "train.json")

        test_path = os.path.join(self.raw_dir, "test.json")

        num_nodes = self.graph.num_nodes
        questions = []
        question_entities_masks = []  # Convert question entities to mask with number of nodes
        supporting_entities_masks = []
        self.num_samples = []
        for path in [train_path, test_path]:
            with open(path) as fin:
                data = json.load(fin)
                self.num_samples.append(len(data))
                for item in data:
                    question = item["question"]
                    questions.append(question)

                    question_entities = list(
                        map(lambda x: self.ent2id[x], item["question_entities"])
                    )
                    question_entities_masks.append(
                        entities_to_mask(question_entities, num_nodes)
                    )  #

                    supporting_entities = list(
                        map(lambda x: self.ent2id[x], item["supporting_entities"])
                    )
                    supporting_entities_masks.append(
                        entities_to_mask(supporting_entities, num_nodes)
                    )

        emb_model = SentenceTransformer(self.emb_model_name)
        self.question_embeddings = emb_model.encode(
            questions, device="cuda", show_progress_bar=True, convert_to_tensor=True
        ).cpu()
        self.question_entities_masks = torch.stack(question_entities_masks)
        self.supporting_entities_masks = torch.stack(supporting_entities_masks)

    def __getitem__(self, index):
        question_emb = self.question_embeddings[index]
        question_entities_mask = self.question_entities_masks[index]
        supporting_entities_mask = self.supporting_entities_masks[index]
        return {
            "question_emb": question_emb,
            "question_entities_mask": question_entities_mask,
            "supporting_entities_mask": supporting_entities_mask,
        }

    def __len__(self):
        return len(self.question_embeddings)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits

    def __repr__(self):
        return f"{self.name}()"


class HotpotQA(QADataset):
    name = "hotpotqa"
