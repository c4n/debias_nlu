from allennlp.models.basic_classifier import BasicClassifier
from typing import Dict, Any, Union, Optional

from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


from torch.autograd import Function

from torch.nn import Module
from torch import tensor
import numpy as np


@Model.register("product_of_expert_basic_classifier")
class ProductofExpertBasicClassifier(BasicClassifier):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Seq2SeqEncoder = None,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, text_field_embedder=text_field_embedder,
                         seq2vec_encoder=seq2vec_encoder, **kwargs)
        # self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        # self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace
        self._namespace = namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(
                namespace=self._label_namespace)

        # Output layer
        self._classification_layer = torch.nn.Linear(
            self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._f1_metrics = [F1Measure(positive_label=i)
                            for i in range(self._num_labels)]

        self._loss = torch.nn.CrossEntropyLoss(reduction='none')
        initializer(self)

    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.IntTensor = None,
        bias_probs: Union[torch.Tensor, np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:

        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(
            tokens)

        if label is not None:

            if bias_probs is not None:
                # output_dict["bert_logits"]=logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                bias_log_probs = torch.log(bias_probs)
                PoE = log_probs + bias_log_probs
                output_dict["logits"] = PoE
                output_dict["probs"] = torch.nn.functional.softmax(PoE, dim=-1)
                PoE = torch.nn.functional.log_softmax(PoE, dim=-1)
                # breakpoint()
                # input for KLDIVLOSS (log_softmax,softmax)
                loss = self._loss(PoE, label.long().view(-1))

            else:
                loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss.mean()
            self._accuracy(logits, label)
            for label_i in range(self._num_labels):
                self._f1_metrics[label_i](probs, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "accuracy": self._accuracy.get_metric(reset),
        }
        for label_i, label_str in self.vocab.get_index_to_token_vocabulary(namespace=self._label_namespace).items():
            for metric, value in self._f1_metrics[label_i].get_metric(reset).items():
                metrics["%s_%s" % (label_str, metric)] = value
        return metrics
