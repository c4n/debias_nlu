"""
The `predict` subcommand allows you to make bulk JSON-to-JSON
or dataset to JSON predictions using a trained model and its
[`Predictor`](../predictors/predictor.md#predictor) wrapper.
"""
from typing import List, Iterator, Optional
import argparse
import sys
import json
from torch import nn, optim

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common import logging as common_logging
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of
from allennlp.data.dataset_readers import MultiTaskDatasetReader
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance

from allennlp.data.dataset_readers import DatasetReader
from my_package.data.dataset_readers.counterfactual_reader import (
    CounterfactualSnliReader,
)
from my_package.data.dataset_readers.counterfactual_reader_mask_ol import (
    CounterfactualSnliReaderMaskOL,
)
from my_package.data.dataset_readers.counterfactual_reader_hypo import (
    CounterfactualSnliHypoReader,
)

from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
    ELMoTokenCharactersIndexer,
    PretrainedTransformerIndexer,
    PretrainedTransformerMismatchedIndexer,
)
from allennlp.data.tokenizers import (
    CharacterTokenizer,
    PretrainedTransformerTokenizer,
    SpacyTokenizer,
    WhitespaceTokenizer,
)

from allennlp.common.util import dump_metrics, sanitize, int_to_device

import pickle

import torch

@Subcommand.register("cf_predict_scale")
class Predict(Subcommand):
    @overrides
    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:

        description = """Run the specified model against a JSON-lines input file."""
        subparser = parser.add_parser(
            self.name,
            description=description,
            help="Use a trained model to make predictions.",
        )

        subparser.add_argument(
            "archive_file", type=str, help="the archived model to make predictions with"
        )
        subparser.add_argument(
            "input_file", type=str, help="path to or url of the input file"
        )
        subparser.add_argument(
            "temperature_file",
            type=str,
            help="path to temperature",
        )

        subparser.add_argument("--output-file", type=str, help="path to output file")
        subparser.add_argument(
            "--weights-file",
            type=str,
            help="a path that overrides which weights file to use",
        )

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="The batch size to use for processing",
        )

        subparser.add_argument(
            "--silent", action="store_true", help="do not print output to stdout"
        )

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
        )

        subparser.add_argument(
            "--use-dataset-reader",
            action="store_true",
            help="Whether to use the dataset reader of the original model to load Instances. "
            "The validation dataset reader will be used if it exists, otherwise it will "
            "fall back to the train dataset reader. This behavior can be overridden "
            "with the --dataset-reader-choice flag.",
        )

        subparser.add_argument(
            "--dataset-reader-choice",
            type=str,
            choices=["train", "validation"],
            default="validation",
            help="Indicates which model dataset reader to use if the --use-dataset-reader "
            "flag is set.",
        )

        subparser.add_argument(
            "--multitask-head",
            type=str,
            default=None,
            help="If you are using a dataset reader to make predictions, and the model is a"
            "multitask model, you have to specify the name of the model head to use here.",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the experiment configuration, e.g., "
                "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.add_argument(
            "--predictor",
            type=str,
            help="optionally specify a specific predictor to use",
        )

        subparser.add_argument(
            "--predictor-args",
            type=str,
            default="",
            help=(
                "an optional JSON structure used to provide additional parameters to the predictor"
            ),
        )

        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.add_argument(
            "--cf_weight",
            type=float,
            default=0.5,
            help="weight for counterfactual component",
        )

        subparser.add_argument(
            "--cf_type",
            type=str,
            default="counterfactual_snli",
            help="counterfactual type",
        )

        subparser.add_argument(
            "--entropy_curve",
            type=float,
            default=0.0,
            help="exponent for entropy",
        )

        subparser.set_defaults(func=_cfpredict)

        return subparser


def _get_cf_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda_device)
    archive = load_archive(
        args.archive_file,
        weights_file=args.weights_file,
        cuda_device=args.cuda_device,
        overrides=args.overrides,
    )
    # Get datareader
    config = archive.config
    model_config = config.get("model")
    model_name = model_config["text_field_embedder"]["token_embedders"]["tokens"][
        "model_name"
    ]
    max_length = model_config["text_field_embedder"]["token_embedders"]["tokens"][
        "max_length"
    ]

    pretrained_transformer_tokenizer = PretrainedTransformerTokenizer(
        model_name=model_name, add_special_tokens=False
    )
    token_indexer = PretrainedTransformerIndexer(
        model_name=model_name, max_length=max_length
    )

    print(args.cf_type)

    dataset_reader = DatasetReader.from_params(
        args.cf_type,
        tokenizer=pretrained_transformer_tokenizer,
        token_indexers={"tokens": token_indexer},
    )
    # if args.cf_type == "mask_all":
    #     dataset_reader = CounterfactualSnliReader(tokenizer=pretrained_transformer_tokenizer,token_indexers={"tokens":token_indexer})
    # elif args.cf_type == "mask_overlap":
    #     dataset_reader = CounterfactualSnliReaderMaskOL(tokenizer=pretrained_transformer_tokenizer,token_indexers={"tokens":token_indexer})

    predictor_args = args.predictor_args.strip()
    if len(predictor_args) <= 0:
        predictor_args = {}
    else:
        import json

        predictor_args = json.loads(predictor_args)
    predictor = Predictor.from_archive(
        archive,
        args.predictor,
        dataset_reader_to_load=args.dataset_reader_choice,
        extra_args=predictor_args,
    )
    predictor._dataset_reader = dataset_reader
    return predictor


class _CFPredictManager:
    def __init__(
        self,
        predictor: Predictor,
        input_file: str,
        temperature_file: str,
        output_file: Optional[str],
        batch_size: int,
        print_to_console: bool,
        has_dataset_reader: bool,
        multitask_head: Optional[str] = None,
        cf_weight: float = 0.0,
        entropy_curve: float = 0.0,
    ) -> None:
        self._predictor = predictor
        self._input_file = input_file
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._temperature = torch.load(temperature_file,map_location=device)
        self._temperature = self._temperature.temperature.to('cpu') # move to cpu
        self._temperature  = self._temperature.detach()
        self._output_file = None if output_file is None else open(output_file, "w")
        self._batch_size = batch_size
        self._print_to_console = print_to_console
        self._dataset_reader = (
            None if not has_dataset_reader else predictor._dataset_reader
        )
        self._cf_weight = cf_weight
        self._entropy_curve = entropy_curve

        self._multitask_head = multitask_head
        if self._multitask_head is not None:
            if self._dataset_reader is None:
                raise ConfigurationError(
                    "You must use a dataset reader when using --multitask-head."
                )
            if not isinstance(self._dataset_reader, MultiTaskDatasetReader):
                raise ConfigurationError(
                    "--multitask-head only works with a multitask dataset reader."
                )
        if (
            isinstance(self._dataset_reader, MultiTaskDatasetReader)
            and self._multitask_head is None
        ):
            raise ConfigurationError(
                "You must specify --multitask-head when using a multitask dataset reader."
            )

    def _predict_json(
        self, batch_data: List[JsonDict], cf_weight: float, entropy_curve: float, temperature : nn.Parameter
    ) -> Iterator[str]:
        if len(batch_data) == 1:
            results = [self._predictor.predict_json(batch_data[0], cf_weight, entropy_curve, self._temperature)]
        else:
            results = self._predictor.predict_batch_json(batch_data, cf_weight, entropy_curve, self._temperature)
        for output in results:
            yield self._predictor.dump_line(output)

    def _predict_instances(self, batch_data: List[Instance], cf_weight: float, entropy_curve: float, temperature : nn.Parameter) -> Iterator[str]:
        if len(batch_data) == 1:
            results = [self._predictor.predict_instance(batch_data[0], cf_weight, entropy_curve, self._temperature)]
        else:
            results = self._predictor.predict_batch_instance(batch_data, cf_weight,entropy_curve, self._temperature)
        for output in results:
            yield self._predictor.dump_line(output)

    def _maybe_print_to_console_and_file(
        self, index: int, prediction: str, model_input: str = None
    ) -> None:
        if self._print_to_console:
            if model_input is not None:
                print(f"input {index}: ", model_input)
            print("prediction: ", prediction)
        if self._output_file is not None:
            self._output_file.write(prediction)

    def _get_json_data(self) -> Iterator[JsonDict]:
        if self._input_file == "-":
            for line in sys.stdin:
                if not line.isspace():
                    yield self._predictor.load_line(line)
        else:
            input_file = cached_path(self._input_file)
            with open(input_file, "r") as file_input:
                for line in file_input:
                    if not line.isspace():
                        yield self._predictor.load_line(line)

    def _get_instance_data(self) -> Iterator[Instance]:
        if self._input_file == "-":
            raise ConfigurationError(
                "stdin is not an option when using a DatasetReader."
            )
        elif self._dataset_reader is None:
            raise ConfigurationError(
                "To generate instances directly, pass a DatasetReader."
            )
        else:
            if isinstance(self._dataset_reader, MultiTaskDatasetReader):
                assert (
                    self._multitask_head is not None
                )  # This is properly checked by the constructor.
                yield from self._dataset_reader.read(
                    self._input_file, force_task=self._multitask_head
                )
            else:
                yield from self._dataset_reader.read(self._input_file)

    def run(self) -> None:
        has_reader = self._dataset_reader is not None
        index = 0
        if has_reader:
            for batch in lazy_groups_of(self._get_instance_data(), self._batch_size):
                for model_input_instance, result in zip(
                    batch, self._predict_instances(batch, self._cf_weight)
                ):
                    self._maybe_print_to_console_and_file(
                        index, result, str(model_input_instance)
                    )
                    index = index + 1
        else:
            for batch_json in lazy_groups_of(self._get_json_data(), self._batch_size):
                for model_input_json, result in zip(
                    batch_json, self._predict_json(batch_json, self._cf_weight, self._entropy_curve, self._temperature)
                ):

                    self._maybe_print_to_console_and_file(
                        index, result, json.dumps(model_input_json)
                    )
                    index = index + 1

        if self._output_file is not None:
            self._output_file.close()


def _cfpredict(args: argparse.Namespace) -> None:
    common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging

    predictor = _get_cf_predictor(args)

    if args.silent and not args.output_file:
        print("--silent specified without --output-file.")
        print("Exiting early because no output will be created.")
        sys.exit(0)

    manager = _CFPredictManager(
        predictor,
        args.input_file,
        args.temperature_file,
        args.output_file,
        args.batch_size,
        not args.silent,
        args.use_dataset_reader,
        args.multitask_head,
        args.cf_weight,
        args.entropy_curve,
    )
    manager.run()
