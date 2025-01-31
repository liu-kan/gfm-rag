from abc import ABC, abstractmethod
from typing import Any


class BaseELModel(ABC):
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def index(self, entity_list: list) -> None:
        """
        Index the given list of entities.

        Args:
            entity_list (list): list of entities

        Returns:
            Any: index of the entities
        """
        pass

    @abstractmethod
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
        pass
