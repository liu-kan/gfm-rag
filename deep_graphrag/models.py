from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data

from deep_graphrag.ultra.models import EntityNBFNet, QueryNBFNet


class SemanticUltra(nn.Module):
    def __init__(
        self, entity_model: EntityNBFNet, rel_emb_dim: int, *args: Any, **kwargs: Any
    ) -> None:
        # kept that because super Ultra sounds cool
        super().__init__()
        self.rel_emb_dim = rel_emb_dim
        self.entity_model = entity_model
        self.rel_mlp = nn.Linear(rel_emb_dim, self.entity_model.dims[0])

    def forward(self, data: Data, batch: torch.Tensor) -> torch.Tensor:
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        batch_size = len(batch)
        relation_representations = (
            self.rel_mlp(data.rel_emb).unsqueeze(0).expand(batch_size, -1, -1)
        )
        h_index, t_index, r_index = batch.unbind(-1)
        # to make NBFNet iteration learn non-trivial paths
        data = self.entity_model.remove_easy_edges(data, h_index, t_index, r_index)
        score = self.entity_model(data, relation_representations, batch)

        return score


class UltraQA(SemanticUltra):
    """Wrap the GNN model for QA."""

    def __init__(
        self, entity_model: QueryNBFNet, rel_emb_dim: int, *args: Any, **kwargs: Any
    ) -> None:
        # kept that because super Ultra sounds cool
        super().__init__(entity_model, rel_emb_dim)
        self.question_mlp = nn.Linear(self.rel_emb_dim, self.entity_model.dims[0])

    def forward(
        self,
        graph: Data,
        batch: torch.Tensor,
        entities_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        question_emb = batch["question_embeddings"]
        question_entities_mask = batch["question_entities_masks"]

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

    def visualize(
        self,
        graph: Data,
        sample: dict[str, torch.Tensor],
        entities_weight: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        question_emb = sample["question_embeddings"]
        question_entities_mask = sample["question_entities_masks"]
        question_embedding = self.question_mlp(question_emb)  # shape: (bs, emb_dim)
        batch_size = question_embedding.size(0)

        assert batch_size == 1, "Currently only supports batch size 1 for visualization"

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
        return self.entity_model.visualize(
            graph,
            sample,
            input,
            relation_representations,
            question_embedding,  # type: ignore
        )
