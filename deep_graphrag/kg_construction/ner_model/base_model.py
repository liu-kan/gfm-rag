from abc import ABC, abstractmethod
from typing import Any


class BaseNERModel(ABC):
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __call__(self, text: str) -> list:
        """
        Perform NER on the given text.

        Args:
            text (str): input text

        Returns:
            list: list of named entities
        """
        pass
