# formerly known as my_fields.py
from typing import Dict

import numpy
import torch
from overrides import overrides

from allennlp.data.fields.field import Field


class FloatField(Field[float]):
    """
    A class representing a scalar float value
    """

    __slots__ = ["value"]

    def __init__(
        self, value: float,
    ) -> None:
        self.value = value

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        return torch.tensor(self.value, dtype=torch.float)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    @overrides
    def empty_field(self):
        return FloatField(-1.0)


    def __str__(self) -> str:
        return f"FloatField with value: {self.value} "

    def __len__(self):
        return 1 

    def __eq__(self, other) -> bool:
        return self.value==other