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
from allennlp.training.metrics import CategoricalAccuracy


from torch.autograd import Function

from torch.nn import Module
from torch import tensor
import numpy as np

            
@Model.register("weighted_distill_basic_classifier")
class WeightedDistillBasicClassifier(BasicClassifier):
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

        super().__init__(vocab, text_field_embedder=text_field_embedder, seq2vec_encoder=seq2vec_encoder, **kwargs)
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
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        
        #Output layer
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss(reduction='none')
        self._distill_loss = torch.nn.KLDivLoss(reduction='batchmean')
        initializer(self)    
        
        
    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.IntTensor = None,
         distill_probs: Union[torch.Tensor, np.ndarray] = None, bias_prob: torch.FloatTensor = None
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
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if label is not None:
            
            if distill_probs is not None and bias_prob is not None:
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                # input for KLDIVLOSS (log_softmax,softmax)
                loss_distill = self._distill_loss(log_probs,distill_probs)
                loss_ce = self._loss(logits, label.long().view(-1))
                loss = loss_ce * (1-bias_prob) + loss_distill * (bias_prob)
                loss = loss * (1/bias_prob)
            else:
                loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss.mean()
            self._accuracy(logits, label)

        return output_dict


# V2 no rewighting at the end
@Model.register("gated_distill_basic_classifier")
class GatedDistillBasicClassifier(BasicClassifier):
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

        super().__init__(vocab, text_field_embedder=text_field_embedder, seq2vec_encoder=seq2vec_encoder, **kwargs)
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
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        
        #Output layer
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss(reduction='none')
        self._distill_loss = torch.nn.KLDivLoss(reduction='batchmean')
        initializer(self)    
        
        
    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.IntTensor = None,
         distill_probs: Union[torch.Tensor, np.ndarray] = None, bias_prob: torch.FloatTensor = None
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
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if label is not None:
            
            if distill_probs is not None and bias_prob is not None:
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                # input for KLDIVLOSS (log_softmax,softmax)
                loss_distill = self._distill_loss(log_probs,distill_probs)
                loss_ce = self._loss(logits, label.long().view(-1))
                loss = loss_ce * (1-bias_prob) + loss_distill * (bias_prob)
                # loss = loss * (1/bias_prob)
            else:
                loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss.mean()
            self._accuracy(logits, label)

        return output_dict