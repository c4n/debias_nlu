import ast
import json
from dataclasses import dataclass
import itertools
from os import PathLike
from typing import Iterable, Iterator, Optional, Union, TypeVar, Dict, List
import logging
import warnings


from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, PretrainedTransformerTokenizer

from allennlp.data.instance import Instance
from allennlp.common import util
from allennlp.common.registrable import Registrable

# add field
from my_package.data.fields.float_fields import FloatField

logger = logging.getLogger(__name__)


PathOrStr = Union[PathLike, str]
DatasetReaderInput = Union[PathOrStr, List[PathOrStr], Dict[str, PathOrStr]]


def maybe_collapse_label(label: str, collapse: bool):
    """
    Helper function that optionally collapses the "contradiction" and "neutral" labels
    into "non-entailment".
    """
    assert label in ["contradiction", "neutral", "entailment"]
    if collapse and label in ["contradiction", "neutral"]:
        return "non-entailment"
    return label


@DatasetReader.register("weighted_fever")
class WeightedFeverReader(DatasetReader):
    """
    Reads a file from the FEVER dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis", along with a metadata field containing the tokenized strings of the
    premise and hypothesis.
    Registered as a `DatasetReader` with name "fever".
    # Parameters
    tokenizer : `Tokenizer`, optional (default=`SpacyTokenizer()`)
        We use this `Tokenizer` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    combine_input_fields : `bool`, optional
            (default=`isinstance(tokenizer, PretrainedTransformerTokenizer)`)
        If False, represent the premise and the hypothesis as separate fields in the instance.
        If True, tokenize them together using `tokenizer.tokenize_sentence_pair()`
        and provide a single `tokens` field in the instance.
    collapse_labels : `bool`, optional (default=`False`)
        If `True`, the "neutral" and "contradiction" labels will be collapsed into "non-entailment";
        "entailment" will be left unchanged.

    Reference: https://fever.ai/dataset/fever.html
    """
    MAP_LABELS = {
        "REFUTES": "contradiction",
        "NOT ENOUGH INFO": "neutral",
        "SUPPORTS": "entailment"
    }

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: Optional[bool] = None,
        collapse_labels: Optional[bool] = False,
        ** kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs
        )
        self._tokenizer = tokenizer or SpacyTokenizer()
        if isinstance(self._tokenizer, PretrainedTransformerTokenizer):
            assert not self._tokenizer._add_special_tokens
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()
        }
        self._combine_input_fields = isinstance(
            self._tokenizer, PretrainedTransformerTokenizer
        )
        self.collapse_labels = collapse_labels

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), 'r') as fh:
            line = fh.readline()
            while line:
                doc = ast.literal_eval(line)

                label_key = "gold_label" if "gold_label" in doc.keys() else "label"
                label = WeightedFeverReader.map_label(doc[label_key])
                evidence_key = "evidence_sentence" if "evidence_sentence" in doc.keys() else "evidence"
                premise = doc[evidence_key]
                hypothesis = doc["claim"]
                sample_weight = doc.get("sample_weight", None)

                yield self.text_to_instance(premise, hypothesis, label, sample_weight)
                line = fh.readline()

    @overrides
    def text_to_instance(
        self,  # type: ignore
        premise: str,
        hypothesis: str,
        label: str = None,
        sample_weight: float = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}
        premise = self._tokenizer.tokenize(premise)
        hypothesis = self._tokenizer.tokenize(hypothesis)

        if self._combine_input_fields:
            tokens = self._tokenizer.add_special_tokens(premise, hypothesis)
            fields["tokens"] = TextField(tokens)
        else:
            premise_tokens = self._tokenizer.add_special_tokens(premise)
            hypothesis_tokens = self._tokenizer.add_special_tokens(hypothesis)
            fields["premise"] = TextField(premise_tokens)
            fields["hypothesis"] = TextField(hypothesis_tokens)

            metadata = {
                "premise_tokens": [x.text for x in premise_tokens],
                "hypothesis_tokens": [x.text for x in hypothesis_tokens],
            }
            fields["metadata"] = MetadataField(metadata)

        if label is not None:
            maybe_collapsed_label = maybe_collapse_label(
                label, self.collapse_labels)
            fields["label"] = LabelField(maybe_collapsed_label)

        if sample_weight is not None:
            fields["sample_weight"] = FloatField(sample_weight)
            
        return Instance(fields)

    @staticmethod
    def map_label(label: str) -> str:
        return WeightedFeverReader.MAP_LABELS[label]

    @overrides
    def apply_token_indexers(self, instance: Instance) -> Instance:
        if "tokens" in instance.fields:
            instance.fields["tokens"]._token_indexers = self._token_indexers
        else:
            instance.fields["premise"]._token_indexers = self._token_indexers
            instance.fields["hypothesis"]._token_indexers = self._token_indexers
