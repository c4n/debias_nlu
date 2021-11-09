from abc import ABC, abstractmethod
from typing import Callable, List, Tuple


FEATURE_EXTRACTOR = Callable[[str, str], float]


class TraditionalML(ABC):
    @abstractmethod
    def fit(self, docs: List[Tuple[str, str]], labels: List[str]) -> None:
        ...

    @abstractmethod
    def inference(self, docs: List[Tuple[str, str]]) -> List[dict]:
        ...

    @abstractmethod
    def predict(self, docs: List[Tuple[str, str]]) -> List[str]:
        ...
