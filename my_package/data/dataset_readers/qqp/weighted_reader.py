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


@DatasetReader.register("weighted_qqp")
class WeightedQQPReader(DatasetReader):
    """
    Reads a file from the QQP dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "is_duplicate", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis", along with a metadata field containing the tokenized strings of the
    premise and hypothesis.
    Registered as a `DatasetReader` with name "qqp".
    # Parameters
    tokenizer : `Tokenizer`, optional (default=`SpacyTokenizer()`)
        We use this `Tokenizer` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.

    Reference: https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs
    """

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: Optional[bool] = None,
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

        if combine_input_fields is not None:
            self._combine_input_fields = combine_input_fields
        else:
            self._combine_input_fields = isinstance(
                self._tokenizer, PretrainedTransformerTokenizer)

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), 'r') as fh:
            line = fh.readline()
            while line:
                doc = json.loads(line)

                label = "paraphrase" if doc["is_duplicate"] else "non-paraphrase"
                premise = doc["sentence1"]
                hypothesis = doc["sentence2"]
                sample_weight = doc.get("sample_weight", None)

                yield self.text_to_instance(premise, hypothesis, label, sample_weight)
                line = fh.readline()

    @overrides
    def text_to_instance(
        self,
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

        fields["label"] = LabelField(label)

        if sample_weight is not None:
            fields["sample_weight"] = FloatField(sample_weight)

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> Instance:
        if "tokens" in instance.fields:
            instance.fields["tokens"]._token_indexers = self._token_indexers
        else:
            instance.fields["premise"]._token_indexers = self._token_indexers
            instance.fields["hypothesis"]._token_indexers = self._token_indexers
