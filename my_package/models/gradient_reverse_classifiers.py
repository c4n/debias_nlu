from allennlp.models.basic_classifier import BasicClassifier
from typing import Dict, Optional

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

class GradientReversalFunction(Function):
    """
    original code: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


            
@Model.register("adv_basic_classifier")
class AdvBasicClassifier(BasicClassifier):
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
        _lambda: float = 1.0,
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
        
        #Gradient Reversal Layer
        self._lambda = _lambda
        self.gradrev = GradientReversal(self._lambda)
        #Output layer
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)    
        
        
    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.IntTensor = None, grad_reverse: bool = False
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
        
        # Reverse Gradient if enabled
        if grad_reverse == True:
            embedded_text = self.gradrev(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

@Model.register("overlap_regressor_classifier")
class OverlapRegressorClassifier(BasicClassifier):
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
        _lambda: float = 1.0,
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
        
        #Gradient Reversal Layer
        self._lambda = _lambda
        self.gradrev = GradientReversal(self._lambda)
        #Output layer
        self._regression_layer = torch.nn.Linear(self._classifier_input_dim, 1)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self._regression_loss = torch.nn.MSELoss()
        initializer(self)    
        
        
    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.IntTensor = None, regression_target: torch.FloatTensor = None, grad_reverse: bool = True
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
        
        # Reverse Gradient if enabled
        if grad_reverse == True:
            embedded_text_rev = self.gradrev(embedded_text)
            # regression outcome
            raw_overlap_score = self._regression_layer(embedded_text_rev)
        else:
            raw_overlap_score = self._regression_layer(embedded_text)            

        # calculate outcome for classifier   
        logits = self._classification_layer(embedded_text)
        
        
        # normalize score
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_overlap_score = F.sigmoid(raw_overlap_score)

        output_dict = {"logits": logits, "probs": probs, "predicted_overlap_score": predicted_overlap_score}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            if regression_target is not None:
                regression_loss = self._regression_loss(regression_target, predicted_overlap_score.float().view(-1)) 
                output_dict["loss"] = loss + regression_loss # todo: separate loss and regression loss later for monitoring purposes
            else:
                output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict