import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn


class RelUltra(nn.Module):
    def __init__(self, entity_model_cfg: DictConfig, rel_emb_dim: int) -> None:
        # kept that because super Ultra sounds cool
        super().__init__()

        self.rel_mlp = nn.Linear(rel_emb_dim, entity_model_cfg["input_dim"])
        self.entity_model = instantiate(entity_model_cfg)

    def forward(self, data, batch):
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        batch_size = len(batch)
        relation_representations = (
            self.rel_mlp(data.rel_emb).unsqueeze(0).expand(batch_size, -1, -1)
        )
        score = self.entity_model(data, relation_representations, batch)

        return score


class UltraQA(RelUltra):
    """Wrap a GNN model for QA."""

    def forward(self, graph, batch):
        question_embedding = self.rel_mlp(
            batch["question_emb"]
        )  # TODO: Use separate mlp for question?
        question_entities_mask = batch["question_entities_mask"]
        batch_size = question_embedding.size(0)
        relation_representations = (
            self.rel_mlp(self.relation_representation)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        # initialize the input with the fuzzy set and question embs
        input = torch.einsum(
            "bn, bd -> bnd", question_entities_mask, question_embedding
        )

        # GNN model: run the entity-level reasoner to get a scalar distribution over nodes
        output = self.entity_model(
            graph, input, relation_representations, question_embedding
        )

        return output
