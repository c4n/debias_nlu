from typing import List, Dict

from overrides import overrides
import numpy
import numpy as np
from copy import deepcopy

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

from allennlp.common.util import dump_metrics, sanitize, int_to_device
from scipy.special import softmax

from my_package.modules.temperature_scaling import *
from torch import nn, optim
import torch

import pickle

def temperature_scale(temperature, logits):
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits
    # temperature = temperature.unsqueeze(1).expand(
    #     logits.size(0), logits.size(1)
    # )
    return logits / temperature


@Predictor.register("cf_textual_entailment_entropy_scale")
class CounterfactualTextualEntailmentPredictor(Predictor):
    """
    Predictor for the [`DecomposableAttention`](../models/decomposable_attention.md) model.
    Registered as a `Predictor` with name "textual_entailment".
    """

    def predict(self, premise: str, hypothesis: str) -> JsonDict:
        """
        Predicts whether the hypothesis is entailed by the premise text.
        # Parameters
        premise : `str`
            A passage representing what is assumed to be true.
        hypothesis : `str`
            A sentence that may be entailed by the premise.
        # Returns
        `JsonDict`
            A dictionary where the key "label_probs" determines the probabilities of each of
            [entailment, contradiction, neutral].
        """
        return self.predict_json({"premise": premise, "hypothesis": hypothesis})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"premise": "...", "hypothesis": "..."}`.
        """
        premise_text = json_dict["premise"]
        hypothesis_text = json_dict["hypothesis"]
        if json_dict["sample_weight"]:
            sample_weight = json_dict["sample_weight"]
            return self._dataset_reader.text_to_instance(
                premise_text, hypothesis_text, sample_weight=sample_weight
            )
        else:
            return self._dataset_reader.text_to_instance(premise_text, hypothesis_text)

    @overrides
    def predict_batch_json(
        self, inputs: List[JsonDict], cf_weight: float, temperature : nn.Parameter
    ) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        return self.predict_batch_instance(instances, cf_weight, temperature)

    @overrides
    def predict_json(self, inputs: JsonDict, cf_weight: float, temperature : nn.Parameter) -> JsonDict:
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance, cf_weight, temperature)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        label = numpy.argmax(outputs["label_logits"])
        # Skip indexing, we have integer representations of the strings "entailment", etc.
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]

    @overrides
    def predict_instance(self, instance: Instance, cf_weight: float) -> JsonDict:
        self._dataset_reader.apply_token_indexers(instance)
        cf_instance = deepcopy(instance)
        cf_instance.fields["tokens"] = cf_instance.fields.pop("cf_tokens")
        instance.fields.pop("cf_tokens")
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(
        self, instances: List[Instance], cf_weight: float,  temperature : nn.Parameter
    ) -> List[JsonDict]:
        cf_instances = []
        label_dict = {0: "entailment", 1: "contradiction", 2: "neutral"}
        for instance in instances:
            self._dataset_reader.apply_token_indexers(instance)
            cf_instance = deepcopy(instance)
            cf_instance.fields["tokens"] = cf_instance.fields.pop("cf_tokens")
            instance.fields.pop("cf_tokens")
            cf_instances.append(cf_instance)

        outputs = self._model.forward_on_instances(instances)
        cf_outputs = self._model.forward_on_instances(cf_instances)
        for i in range(len(outputs)):
            # temperature = temperature.to('cpu')
            outputs[i]["logits"] = temperature_scale(temperature,outputs[i]["logits"])
            cf_outputs[i]["logits"] = temperature_scale(temperature,cf_outputs[i]["logits"])
            n_class = outputs[i]["logits"].shape[0]
            factual_softmax = softmax(outputs[i]["logits"])
            factual_entropy = -sum(
                factual_softmax * np.log(factual_softmax) / np.log(n_class)
            )
            # outputs[i]['logits'] = outputs[i]['logits'] -  (factual_entropy  * cf_weight * cf_outputs[i]['logits'])
            # outputs[i]['logits'] = outputs[i]['logits'] -  cf_weight * cf_outputs[i]['logits']
            outputs[i]["probs"] = softmax(outputs[i]["logits"]) - (
                (factual_entropy) * cf_weight * softmax(cf_outputs[i]["logits"])
            )
            outputs[i]["label"] = label_dict[np.argmax(outputs[i]["probs"]).item()]
        return sanitize(outputs)
