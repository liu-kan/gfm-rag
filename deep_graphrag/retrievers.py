from abc import ABC, abstractmethod

import torch


class BaseRetriever(ABC):
    def __init__(self, docs: dict, id2doc: dict) -> None:
        self.docs = docs
        self.id2doc = id2doc

    @abstractmethod
    def __call__(self, doc_ranking: torch.Tensor, top_k: int = 1) -> list:
        pass


class SimpleRetriever(BaseRetriever):
    def __call__(self, doc_ranking: torch.Tensor, top_k: int = 1) -> list:
        top_k_docs = doc_ranking.topk(top_k).indices
        return [
            {
                "title": self.id2doc[doc.item()],
                "content": self.docs[self.id2doc[doc.item()]],
            }
            for doc in top_k_docs
        ]
