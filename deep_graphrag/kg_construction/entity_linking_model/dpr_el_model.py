import hashlib
import os

import torch
from sentence_transformers import SentenceTransformer

from .base_model import BaseELModel


class DPRELModel(BaseELModel):
    def __init__(
        self,
        model_name: str,
        root: str = "tmp",
        use_cache: bool = True,
        normalize: bool = True,
        query_instruct: str = "",
        model_kwargs: dict | None = None,
    ) -> None:
        self.model_name = model_name
        self.use_cache = use_cache
        self.normalize = normalize
        self.root = os.path.join(root, f"{self.model_name.split("/")[0]}_dpr_cache")
        if self.use_cache and not os.path.exists(self.root):
            os.makedirs(self.root)
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, model_kwargs=model_kwargs
        )
        self.query_instruct = query_instruct

    def index(self, entity_list: list) -> None:
        self.entity_list = entity_list
        # Get md5 fingerprint of the whole given entity list
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        cache_file = f"{self.root}/{fingerprint}.pt"
        if os.path.exists(cache_file):
            self.entity_embeddings = torch.load(cache_file)
        else:
            self.entity_embeddings = self.model.encode(
                entity_list,
                device="cuda" if torch.cuda.is_available() else "cpu",
                convert_to_tensor=True,
                show_progress_bar=True,
                normalize_embeddings=self.normalize,
            )
            if self.use_cache:
                torch.save(self.entity_embeddings, cache_file)

    def __call__(self, ner_entity_list: list) -> list:
        ner_entity_embeddings = self.model.encode(
            [self.query_instruct + q for q in ner_entity_list],
            device="cuda" if torch.cuda.is_available() else "cpu",
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
        )
        scores = ner_entity_embeddings @ self.entity_embeddings.T
        linked_entity_list = []
        for i in range(len(ner_entity_list)):
            linked_entity_list.append(self.entity_list[scores[i].argmax().item()])
        return linked_entity_list
