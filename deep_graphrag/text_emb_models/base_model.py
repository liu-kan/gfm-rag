import torch
from sentence_transformers import SentenceTransformer


class BaseTextEmbModel:
    def __init__(
        self,
        text_emb_model_name: str,
        normalize: bool,
        batch_size: int,
        query_instruct: str = "",
        passage_instruct: str = "",
        model_kwargs: dict | None = None,
    ) -> None:
        self.text_emb_model_name = text_emb_model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.query_instruct = query_instruct
        self.passage_instruct = passage_instruct
        self.model_kwargs = model_kwargs

        self.text_emb_model = SentenceTransformer(
            self.text_emb_model_name,
            trust_remote_code=True,
            model_kwargs=self.model_kwargs,
        )

    def encode(self, text: list[str], is_query: bool = False) -> torch.Tensor:
        return self.text_emb_model.encode(
            text,
            device="cuda" if torch.cuda.is_available() else "cpu",
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
            prompt=self.query_instruct if is_query else self.passage_instruct,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
