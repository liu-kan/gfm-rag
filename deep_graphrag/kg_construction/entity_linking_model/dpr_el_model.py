import hashlib
import os
from typing import Any

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
        batch_size: int = 32,
        query_instruct: str = "",
        passage_instruct: str = "",
        model_kwargs: dict | None = None,
    ) -> None:
        self.model_name = model_name
        self.use_cache = use_cache
        self.normalize = normalize
        self.batch_size = batch_size
        self.root = os.path.join(
            root, f"{self.model_name.replace("/", "_")[0]}_dpr_cache"
        )
        if self.use_cache and not os.path.exists(self.root):
            os.makedirs(self.root)
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, model_kwargs=model_kwargs
        )
        self.query_instruct = query_instruct
        self.passage_instruct = passage_instruct

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
                prompt=self.passage_instruct,
                normalize_embeddings=self.normalize,
                batch_size=self.batch_size,
            )
            if self.use_cache:
                torch.save(self.entity_embeddings, cache_file)

    def __call__(self, ner_entity_list: list) -> list:
        ner_entity_embeddings = self.model.encode(
            ner_entity_list,
            device="cuda" if torch.cuda.is_available() else "cpu",
            convert_to_tensor=True,
            prompt=self.query_instruct,
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
        )
        scores = ner_entity_embeddings @ self.entity_embeddings.T
        linked_entity_list = []
        for i in range(len(ner_entity_list)):
            linked_entity_list.append(self.entity_list[scores[i].argmax().item()])
        return linked_entity_list


class NVEmbedV2ELModel(DPRELModel):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"

    def add_eos(self, input_examples: list[str]) -> list[str]:
        input_examples = [
            input_example + self.model.tokenizer.eos_token
            for input_example in input_examples
        ]
        return input_examples

    def __call__(self, ner_entity_list: list) -> list:
        ner_entity_list = self.add_eos(ner_entity_list)
        return super().__call__(ner_entity_list)
