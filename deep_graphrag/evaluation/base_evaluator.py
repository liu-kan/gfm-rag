import json
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    def __init__(self, prediction_file: str) -> None:
        super().__init__()
        with open(prediction_file) as f:
            self.data = [json.loads(line) for line in f]

    @abstractmethod
    def evaluate(self) -> dict:
        pass
