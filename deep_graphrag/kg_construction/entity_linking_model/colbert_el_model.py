import hashlib
import os
import re

from colbert import Indexer, Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig

from .base_model import BaseELModel


def processing_phrases(phrase: str) -> str:
    return re.sub("[^A-Za-z0-9 ]", " ", phrase.lower()).strip()


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

    def __call__(self, ner_entity_list: list) -> list:
        """
        Link entities in the given text to the knowledge graph.

        Args:
            ner_entity_list (list): list of named entities

        Returns:
            list: list of linked entities in the knowledge graph
        """

        try:
            self.__getattribute__("phrase_searcher")
        except AttributeError as e:
            raise AttributeError("Index the entities first using index method") from e

        ner_entity_list = [processing_phrases(p) for p in ner_entity_list]
        phrase_ids = []
        for query in ner_entity_list:
            queries = Queries(path=None, data={0: query})
            ranking = self.phrase_searcher.search_all(queries, k=1)

            for phrase_id, _rank, _score in ranking.data[0]:
                phrase_ids.append(phrase_id)

        linked_entity_list = [self.entity_list[phrase_id] for phrase_id in phrase_ids]

        return linked_entity_list
