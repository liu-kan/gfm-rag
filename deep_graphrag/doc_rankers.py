from abc import ABC, abstractmethod

import torch
from scipy import sparse


class BaseDocRanker(ABC):
    """
    Abstract class for document ranker

    Args:
        ent2doc (torch.Tensor): Mapping from entity to document
    """

    def __init__(self, ent2doc: torch.Tensor) -> None:
        self.ent2doc = ent2doc

    @abstractmethod
    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        pass


class SimpleRanker(BaseDocRanker):
    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on entity prediction

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """
        doc_pred = torch.sparse.mm(ent_pred, self.ent2doc)
        return doc_pred


class IDFWeightedRanker(BaseDocRanker):
    """
    Rank documents based on entity prediction with IDF weighting
    """

    def __init__(self, ent2doc: torch.Tensor) -> None:
        super().__init__(ent2doc)
        self.idf_weight = 1 / ent2doc.to_dense().sum(dim=-1)
        self.idf_weight[ent2doc.to_dense().sum(dim=-1) == 0] = 0

    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on entity prediction with IDF weighting

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """
        doc_pred = torch.sparse.mm(
            ent_pred * self.idf_weight.unsqueeze(0), self.ent2doc
        )
        return doc_pred


class TopKRanker(BaseDocRanker):
    def __init__(self, ent2doc: torch.Tensor, top_k: int) -> None:
        super().__init__(ent2doc)
        self.top_k = top_k

    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on top-k entity prediction

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """
        top_k_ent_pred = torch.topk(ent_pred, self.top_k, dim=-1)
        masked_ent_pred = torch.zeros_like(ent_pred, device=ent_pred.device)
        masked_ent_pred.scatter_(1, top_k_ent_pred.indices, 1)
        doc_pred = torch.sparse.mm(masked_ent_pred, self.ent2doc)
        return doc_pred


class IDFWeightedTopKRanker(BaseDocRanker):
    def __init__(self, ent2doc: torch.Tensor, top_k: int) -> None:
        super().__init__(ent2doc)
        self.top_k = top_k
        self.idf_weight = 1 / ent2doc.to_dense().sum(dim=-1)
        self.idf_weight[ent2doc.to_dense().sum(dim=-1) == 0] = 0

    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on top-k entity prediction

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """
        top_k_ent_pred = torch.topk(ent_pred, self.top_k, dim=-1)
        idf_weight = torch.gather(
            self.idf_weight.expand(ent_pred.shape[0], -1), 1, top_k_ent_pred.indices
        )
        masked_ent_pred = torch.zeros_like(ent_pred, device=ent_pred.device)
        masked_ent_pred.scatter_(1, top_k_ent_pred.indices, idf_weight)
        doc_pred = torch.sparse.mm(masked_ent_pred, self.ent2doc)
        return doc_pred


class TFIDFWeightedRanker(BaseDocRanker):
    """
    Rank documents based on entity prediction with IDF weighting
    """

    def __init__(self, ent2doc: torch.Tensor) -> None:
        super().__init__(ent2doc)
        self.idf_weight = torch.log(
            ent2doc.shape[1] / (ent2doc.to_dense().sum(dim=-1) + 1)
        )
        tf_weight = sparse.load_npz(
            "data/hotpotqa/processed/stage2/tf_count.npz"
        )  # (num_entities, num_docs)
        self.tf_weight = torch.tensor(tf_weight.toarray(), dtype=torch.float32).to(
            self.ent2doc.device
        )
        # self.tf_weight = self.tf_weight / self.tf_weight.sum(-1, keepdim=True)
        self.tfidf_weight = self.tf_weight * self.idf_weight.unsqueeze(-1)

    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on entity prediction with IDF weighting

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """

        doc_pred = torch.matmul(ent_pred.softmax(dim=-1), self.tfidf_weight)
        # doc_pred = torch.matmul(ent_pred.softmax(dim=-1), self.tf_weight)

        return doc_pred


class TFIDFWeightedTopKRanker(BaseDocRanker):
    def __init__(self, ent2doc: torch.Tensor, top_k: int) -> None:
        super().__init__(ent2doc)
        self.ent2doc = ent2doc
        self.top_k = top_k
        self.idf_weight = torch.log(
            ent2doc.shape[1] / (ent2doc.to_dense().sum(dim=-1) + 1)
        )
        # Set zero occurence to ent2doc.to_dense().sum(dim=-1) == 0
        self.idf_weight[ent2doc.to_dense().sum(dim=-1) == 0] = 0

        tf_weight = sparse.load_npz(
            "data/hotpotqa/processed/stage2/tf_count.npz"
        )  # (num_entities, num_docs)
        self.tf_weight = torch.tensor(tf_weight.toarray(), dtype=torch.float32).to(
            self.ent2doc.device
        )
        # self.tf_weight = self.tf_weight / self.tf_weight.sum(-1, keepdim=True)
        self.tfidf_weight = self.tf_weight * self.idf_weight.unsqueeze(-1)

    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on top-k entity prediction

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """
        top_k_ent_pred = torch.topk(ent_pred, self.top_k, dim=-1)

        tfidf_weight_ = self.tfidf_weight.expand(ent_pred.shape[0], -1, -1)
        top_k_ent_pred_ = top_k_ent_pred.indices.unsqueeze(-1).expand(
            -1, -1, self.tfidf_weight.shape[-1]
        )
        tfidf_weight = torch.gather(tfidf_weight_, 1, top_k_ent_pred_)
        masked_ent_pred = torch.zeros_like(tfidf_weight_, device=tfidf_weight_.device)
        masked_ent_pred.scatter_(1, top_k_ent_pred_, tfidf_weight)

        doc_pred = (ent_pred.softmax(dim=-1).unsqueeze(-1) * masked_ent_pred).sum(dim=1)

        return doc_pred
