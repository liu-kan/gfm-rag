import json
import logging
import os
from collections.abc import Callable

import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data, InMemoryDataset, download_url

from deep_graphrag.ultra.tasks import build_relation_graph

logger = logging.getLogger(__name__)


class KGDataset(InMemoryDataset):
    delimiter = "\t"

    def __init__(
        self,
        root: str,
        data_name: str,
        text_emb_model_name: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = build_relation_graph,
        **kwargs: str,
    ) -> None:
        self.name = data_name
        self.emb_model_name = text_emb_model_name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> list:
        return ["kg.txt"]

    def download(self) -> None:
        """Download the dataset."""

        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url, self.raw_dir)
            os.rename(download_path, path)

    def load_file(
        self, triplet_file: str, inv_entity_vocab: dict, inv_rel_vocab: dict
    ) -> dict:
        """Load a knowledge graph file and return the processed data."""

        triplets = []  # Triples with inverse relations
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, encoding="utf-8") as fin:
            for line in fin:
                try:
                    u, r, v = (
                        line.split()
                        if self.delimiter is None
                        else line.strip().split(self.delimiter)
                    )
                except Exception as e:
                    logger.error(f"Error in line: {line}, {e}, Skipping")
                    continue
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab),  # entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab,
        }

    def process(self) -> None:
        kg_file = self.raw_paths[0]

        kg_result = self.load_file(kg_file, inv_entity_vocab={}, inv_rel_vocab={})

        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train adn 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = kg_result["num_node"]
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = kg_result["num_relation"]

        kg_triplets = kg_result["triplets"]

        train_target_edges = torch.tensor(
            [[t[0], t[1]] for t in kg_triplets], dtype=torch.long
        ).t()
        train_target_etypes = torch.tensor([t[2] for t in kg_triplets])

        # Add inverse edges
        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat(
            [train_target_etypes, train_target_etypes + num_relations]
        )

        with open(self.processed_dir + "/ent2id.json", "w") as f:
            json.dump(kg_result["inv_entity_vocab"], f)
        rel2id = kg_result["inv_rel_vocab"]
        id2rel = {v: k for k, v in rel2id.items()}
        for etype in train_etypes:
            if etype.item() >= num_relations:
                raw_etype = etype - num_relations
                raw_rel = id2rel[raw_etype.item()]
                rel2id["inverse_" + raw_rel] = etype.item()
        with open(self.processed_dir + "/rel2id.json", "w") as f:
            json.dump(rel2id, f)

        # Generate relation embeddings
        logger.info("Generating relation embeddings")
        text_emb_model = SentenceTransformer(self.emb_model_name)
        rel_emb = text_emb_model.encode(
            list(rel2id.keys()),
            device="cuda" if torch.cuda.is_available() else "cpu",
            show_progress_bar=True,
            convert_to_tensor=True,
        ).cpu()

        kg_data = Data(
            edge_index=train_edges,
            edge_type=train_etypes,
            num_nodes=num_node,
            target_edge_index=train_target_edges,
            target_edge_type=train_target_etypes,
            num_relations=num_relations * 2,
            rel_emb=rel_emb,
        )

        # build graphs of relations
        if self.pre_transform is not None:
            kg_data = self.pre_transform(kg_data)

        torch.save((self.collate([kg_data])), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name}()"

    @property
    def num_relations(self) -> int:
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self) -> str:
        return os.path.join(str(self.root), str(self.name), "processed", "stage1")

    @property
    def processed_dir(self) -> str:
        return os.path.join(str(self.root), str(self.name), "processed", "stage2")

    @property
    def processed_file_names(self) -> str:
        return "data.pt"
