import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from gfmrag import utils
from gfmrag.datasets import QADataset
from gfmrag.doc_rankers import BaseDocRanker
from gfmrag.kg_construction.entity_linking_model import BaseELModel
from gfmrag.kg_construction.ner_model import BaseNERModel
from gfmrag.models import GNNRetriever
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.ultra import query_utils
from gfmrag.utils.qa_utils import entities_to_mask

logger = logging.getLogger(__name__)


class GFMRetriever:
    def __init__(
        self,
        qa_data: QADataset,
        text_emb_model: BaseTextEmbModel,
        ner_model: BaseNERModel,
        el_model: BaseELModel,
        graph_retriever: GNNRetriever,
        doc_ranker: BaseDocRanker,
        doc_retriever: utils.DocumentRetriever,
        entities_weight: torch.Tensor | None,
        device: torch.device,
    ) -> None:
        self.qa_data = qa_data
        self.graph = qa_data.kg
        self.text_emb_model = text_emb_model
        self.ner_model = ner_model
        self.el_model = el_model
        self.graph_retriever = graph_retriever
        self.doc_ranker = doc_ranker
        self.doc_retriever = doc_retriever
        self.device = device
        self.num_nodes = self.graph.num_nodes
        self.entities_weight = entities_weight

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """
        Retrieve documents from the corpus based on the given query.

        Args:
            query (str): input query
            top_k (int): number of documents to retrieve

        Returns:
            list: list of retrieved documents
        """

        # Prepare input for deep graph retriever
        graph_retriever_input = self.prepare_input_for_graph_retriever(query)
        graph_retriever_input = query_utils.cuda(
            graph_retriever_input, device=self.device
        )

        # Graph retriever forward pass
        ent_pred = self.graph_retriever(
            self.graph, graph_retriever_input, entities_weight=self.entities_weight
        )
        doc_pred = self.doc_ranker(ent_pred)[0]  # Ent2docs mapping, batch size is 1

        # Retrieve the supporting documents
        retrieved_docs = self.doc_retriever(doc_pred.cpu(), top_k=top_k)

        return retrieved_docs

    def prepare_input_for_graph_retriever(self, query: str) -> dict:
        """
        Prepare input for the graph retriever model.

        Args:
            query (str): input query

        Returns:
            dict: input for the graph retriever model
        """

        # Prepare input for deep graph retriever
        mentioned_entities = self.ner_model(query)
        if len(mentioned_entities) == 0:
            logger.warning(
                "No mentioned entities found in the query. Use the query as is for entity linking."
            )
            mentioned_entities = [query]
        linked_entities = self.el_model(mentioned_entities, topk=1)
        entity_ids = [
            self.qa_data.ent2id[ent[0]["entity"]]
            for ent in linked_entities.values()
            if ent[0]["entity"] in self.qa_data.ent2id
        ]
        question_entities_masks = (
            entities_to_mask(entity_ids, self.num_nodes).unsqueeze(0).to(self.device)
        )  # 1 x num_nodes
        question_embedding = self.text_emb_model.encode(
            [query],
            is_query=True,
            show_progress_bar=False,
        )
        graph_retriever_input = {
            "question_embeddings": question_embedding,
            "question_entities_masks": question_entities_masks,
        }
        return graph_retriever_input

    @staticmethod
    def from_config(cfg: DictConfig) -> "GFMRetriever":
        graph_retriever, model_config = utils.load_model_from_pretrained(
            cfg.graph_retriever.model_path
        )
        graph_retriever.eval()
        qa_data = QADataset(
            **cfg.dataset,
            text_emb_model_cfgs=OmegaConf.create(model_config["text_emb_model_config"]),
        )
        device = utils.get_device()
        graph_retriever = graph_retriever.to(device)

        qa_data.kg = qa_data.kg.to(device)
        ent2docs = qa_data.ent2docs.to(device)

        ner_model = instantiate(cfg.graph_retriever.ner_model)
        el_model = instantiate(cfg.graph_retriever.el_model)

        el_model.index(list(qa_data.ent2id.keys()))

        # Create doc ranker
        doc_ranker = instantiate(cfg.graph_retriever.doc_ranker, ent2doc=ent2docs)
        doc_retriever = utils.DocumentRetriever(qa_data.doc, qa_data.id2doc)

        text_emb_model = instantiate(
            OmegaConf.create(model_config["text_emb_model_config"])
        )

        entities_weight = None
        if cfg.graph_retriever.init_entities_weight:
            entities_weight = utils.get_entities_weight(ent2docs)

        return GFMRetriever(
            qa_data=qa_data,
            text_emb_model=text_emb_model,
            ner_model=ner_model,
            el_model=el_model,
            graph_retriever=graph_retriever,
            doc_ranker=doc_ranker,
            doc_retriever=doc_retriever,
            entities_weight=entities_weight,
            device=device,
        )
