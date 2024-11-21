from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch_geometric.data import Data


class SemanticUltra(nn.Module):
    def __init__(
        self, entity_model_cfg: DictConfig, rel_emb_dim: int, *args: Any, **kwargs: Any
    ) -> None:
        # kept that because super Ultra sounds cool
        super().__init__()
        self.rel_emb_dim = rel_emb_dim
        self.rel_mlp = nn.Linear(rel_emb_dim, entity_model_cfg["input_dim"])
        self.entity_model = instantiate(entity_model_cfg)

    def forward(self, data: Data, batch: torch.Tensor) -> torch.Tensor:
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        batch_size = len(batch)
        relation_representations = (
            self.rel_mlp(data.rel_emb).unsqueeze(0).expand(batch_size, -1, -1)
        )
        score = self.entity_model(data, relation_representations, batch)

        return score


class UltraQA(SemanticUltra):
    """Wrap the GNN model for QA."""

    def __init__(
        self, entity_model_cfg: DictConfig, rel_emb_dim: int, *args: Any, **kwargs: Any
    ) -> None:
        # kept that because super Ultra sounds cool
        super().__init__(entity_model_cfg, rel_emb_dim)
        self.question_mlp = nn.Linear(self.rel_emb_dim, entity_model_cfg["input_dim"])

    def forward(
        self,
        graph: Data,
        batch: torch.Tensor,
        entities_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        question_emb = batch[0]
        question_entities_mask = batch[1]

        question_embedding = self.question_mlp(question_emb)  # shape: (bs, emb_dim)
        batch_size = question_embedding.size(0)
        relation_representations = (
            self.rel_mlp(graph.rel_emb).unsqueeze(0).expand(batch_size, -1, -1)
        )

        # initialize the input with the fuzzy set and question embs
        if entities_weight is not None:
            question_entities_mask = question_entities_mask * entities_weight.unsqueeze(
                0
            )

        input = torch.einsum(
            "bn, bd -> bnd", question_entities_mask, question_embedding
        )

        # GNN model: run the entity-level reasoner to get a scalar distribution over nodes
        output = self.entity_model(
            graph, input, relation_representations, question_embedding
        )

        return output
