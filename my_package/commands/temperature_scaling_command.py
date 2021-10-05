# TO DO: test gpu

import torch
from torch import nn, optim
from torch.nn import functional as F

from allennlp.models.archival import load_archive
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.params import Params
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import dump_metrics, sanitize, int_to_device
from allennlp.data import Instance, Vocabulary, Batch, DataLoader
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models.archival import CONFIG_NAME
from allennlp.models.model import Model
from allennlp.nn import util as nn_util

import pickle
from my_package.modules import temperature_scaling
from my_package.modules.temperature_scaling import *

import argparse
from allennlp.commands.subcommand import Subcommand
from allennlp.common import logging as common_logging
from allennlp.common.util import prepare_environment

from overrides import overrides

logger = logging.getLogger(__name__)

@Subcommand.register("temp_scale")
class TemperatureScaledModel(Subcommand):
    @overrides
    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        description = """Get temperature  scaled model"""
        subparser = parser.add_parser(
            self.name, description=description, help="Get temperature  scaled model."
        )

        subparser.add_argument(
            "archive_file", type=str, help="path to an archived trained model"
        )

        subparser.add_argument(
            "input_file",
            type=str,
            help="path to the file containing the evaluation data",
        )

        subparser.add_argument(
            "--output-file",
            type=str,
            help="optional path to write the metrics to as JSON",
        )

        subparser.add_argument(
            "--weights-file",
            type=str,
            help="a path that overrides which weights file to use",
        )

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
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
            "--batch-size",
            type=int,
            help="If non-empty, the batch size to use during evaluation.",
        )

        subparser.add_argument(
            "--batch-weight-key",
            type=str,
            default="",
            help="If non-empty, name of metric used to weight the loss on a per-batch basis.",
        )

        subparser.add_argument(
            "--extend-vocab",
            action="store_true",
            default=False,
            help="if specified, we will use the instances in your new dataset to "
            "extend your vocabulary. If pretrained-file was used to initialize "
            "embedding layers, you may also need to pass --embedding-sources-mapping.",
        )

        subparser.add_argument(
            "--embedding-sources-mapping",
            type=str,
            default="",
            help="a JSON dict defining mapping from embedding module path to embedding "
            "pretrained-file used during training. If not passed, and embedding needs to be "
            "extended, we will try to use the original file paths used during training. If "
            "they are not available we will use random vectors for embedding extension.",
        )
        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.set_defaults(func=temp_scale_from_args)

        return subparser


def temp_scale_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging

    # Disable some of the more verbose logging statements
    logging.getLogger("allennlp.common.params").disabled = True
    logging.getLogger("allennlp.nn.initializers").disabled = True
    logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(
        logging.INFO
    )

    # Load from archive
    archive = load_archive(
        args.archive_file,
        weights_file=args.weights_file,
        cuda_device=args.cuda_device,
        overrides=args.overrides,
    )
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data
    dataset_reader = archive.validation_dataset_reader

    # output file
    output_file_path = None
    predictions_output_file_path = None

    evaluation_data_path = args.input_file
    if args.output_file != None:
        output_file_path = args.output_file

    logger.info("Reading evaluation data from %s", evaluation_data_path)
    data_loader_params = config.get("validation_data_loader", None)
    if data_loader_params is None:
        data_loader_params = config.get("data_loader")
    if args.batch_size:
        data_loader_params["batch_size"] = args.batch_size
    data_loader = DataLoader.from_params(
        params=data_loader_params, reader=dataset_reader, data_path=evaluation_data_path
    )

    embedding_sources = (
        json.loads(args.embedding_sources_mapping)
        if args.embedding_sources_mapping
        else {}
    )

    if args.extend_vocab:
        logger.info("Vocabulary is being extended with test instances.")
        model.vocab.extend_from_instances(instances=data_loader.iter_instances())
        model.extend_embedder_vocab(embedding_sources)

    data_loader.index_with(model.vocab)

    temp_scaled_model = get_temp(
        model, data_loader, args.cuda_device, output_file=output_file_path
    )
    logger.info("Finished evaluating.")

    return temp_scaled_model.temperature
