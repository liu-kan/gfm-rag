import torch

from .base_model import BaseTextEmbModel


class NVEmbedV2(BaseTextEmbModel):
    def __init__(
        self,
        text_emb_model_name: str,
        normalize: bool,
        batch_size: int,
        query_instruct: str = "",
        passage_instruct: str = "",
        model_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            text_emb_model_name,
            normalize,
            batch_size,
            query_instruct,
            passage_instruct,
            model_kwargs,
        )
        self.text_emb_model.max_seq_length = 32768
        self.text_emb_model.tokenizer.padding_side = "right"

    def add_eos(self, input_examples: list[str]) -> list[str]:
        input_examples = [
            input_example + self.text_emb_model.tokenizer.eos_token
            for input_example in input_examples
        ]
        return input_examples

    def encode(self, text: list[str], is_query: bool = False) -> torch.Tensor:
        return super().encode(self.add_eos(text), is_query)
