import hashlib
import os

from colbert import Indexer, Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig

from deep_graphrag.kg_construction.utils import processing_phrases

from .base_model import BaseELModel


class ColbertELModel(BaseELModel):
    def __init__(
        self,
        checkpint_path: str,
        root: str = "tmp",
        doc_index_name: str = "nbits_2",
        phrase_index_name: str = "nbits_2",
    ) -> None:
        if not os.path.exists(checkpint_path):
            raise FileNotFoundError(
                "Checkpoint not found, download the checkpoint with: 'wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz'"
            )
        self.checkpoint_path = checkpint_path
        self.root = root
        self.doc_index_name = doc_index_name
        self.phrase_index_name = phrase_index_name

    def index(self, entity_list: list) -> None:
        self.entity_list = entity_list
        # Get md5 fingerprint of the whole given entity list
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        exp_name = f"Entity_index_{fingerprint}"
        colbert_config = {
            "root": f"{self.root}/colbert/{fingerprint}",
            "doc_index_name": self.doc_index_name,
            "phrase_index_name": self.phrase_index_name,
        }
        phrases = [processing_phrases(p) for p in entity_list]
        with Run().context(
            RunConfig(nranks=1, experiment=exp_name, root=colbert_config["root"])
        ):
            config = ColBERTConfig(
                nbits=2,
                root=colbert_config["root"],
            )
            indexer = Indexer(checkpoint=self.checkpoint_path, config=config)
            indexer.index(
                name=self.phrase_index_name, collection=phrases, overwrite="reuse"
            )

        with Run().context(
            RunConfig(nranks=1, experiment=exp_name, root=colbert_config["root"])
        ):
            config = ColBERTConfig(
                root=colbert_config["root"],
            )
            phrase_searcher = Searcher(
                index=colbert_config["phrase_index_name"], config=config, verbose=1
            )
        self.phrase_searcher = phrase_searcher

    def __call__(self, ner_entity_list: list, topk: int = 1) -> dict:
        """
        Link entities in the given text to the knowledge graph.

        Args:
            ner_entity_list (list): list of named entities
            topk (int): number of linked entities to return

        Returns:
            dict: dict of linked entities in the knowledge graph
                key (str): named entity
                value (list[dict]): list of linked entities
                    entity: linked entity
                    score: score of the entity
                    norm_score: normalized score of the entity

        """

        try:
            self.__getattribute__("phrase_searcher")
        except AttributeError as e:
            raise AttributeError("Index the entities first using index method") from e

        ner_entity_list = [processing_phrases(p) for p in ner_entity_list]
        query_data: dict[int, str] = {
            i: query for i, query in enumerate(ner_entity_list)
        }

        queries = Queries(path=None, data=query_data)
        ranking = self.phrase_searcher.search_all(queries, k=topk)

        linked_entity_dict: dict[str, list] = {}
        for i in range(len(queries)):
            query = queries[i]
            rank = ranking.data[i]
            linked_entity_dict[query] = []
            max_score = rank[0][2]

            for phrase_id, _rank, score in rank:
                linked_entity_dict[query].append(
                    {
                        "entity": self.entity_list[phrase_id],
                        "score": score,
                        "norm_score": score / max_score,
                    }
                )

        return linked_entity_dict
