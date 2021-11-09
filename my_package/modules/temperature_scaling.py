# TO DO: test gpu
from typing import Dict, Any, Union, Optional

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


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        output_dict = self.model(input)
        logits = output_dict["logits"]
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            self.model
            for input in valid_loader:
                input = input
                label = input["label"].cuda()
                output_dict = self.model(tokens=input["tokens"], label=label)
                logits = output_dict["logits"]
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print(
            "Before temperature - NLL: %.3f, ECE: %.3f"
            % (before_temperature_nll, before_temperature_ece)
        )

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(
            self.temperature_scale(logits), labels
        ).item()
        after_temperature_ece = ece_criterion(
            self.temperature_scale(logits), labels
        ).item()
        print("Optimal temperature: %.3f" % self.temperature.item())
        print(
            "After temperature - NLL: %.3f, ECE: %.3f"
            % (after_temperature_nll, after_temperature_ece)
        )

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def get_temp(
    model: Model,
    data_loader: DataLoader,
    cuda_device: Union[int, torch.device] = -1,
    output_file: str = None,
) -> Dict[str, Any]:
    """
    # Parameters
    model : `Model`
        The model to evaluate
    data_loader : `DataLoader`
        The `DataLoader` that will iterate over the evaluation data (data loaders already contain
        their data).
    cuda_device : `Union[int, torch.device]`, optional (default=`-1`)
        The cuda device to use for this evaluation.  The model is assumed to already be using this
        device; this parameter is only used for moving the input data to the correct device.
    batch_weight_key : `str`, optional (default=`None`)
        If given, this is a key in the output dictionary for each batch that specifies how to weight
        the loss for that batch.  If this is not given, we use a weight of 1 for every batch.
    metrics_output_file : `str`, optional (default=`None`)
        Optional path to write the final metrics to.
    predictions_output_file : `str`, optional (default=`None`)
        Optional path to write the predictions to.
    # Returns
    `Dict[str, Any]`
        The final metrics.
    """
    check_for_gpu(cuda_device)
    data_loader.set_target_device(int_to_device(cuda_device))
    model_temp = ModelWithTemperature(model)
    model_temp.set_temperature(data_loader)
    # pickle.dump(model_temp, open(output_file, "wb"))
    torch.save(model_temp,output_file)
    return model_temp


# @Subcommand.register("temp_scale")
# class TempuratureScaledModel(Subcommand):
#     @overrides
#     def add_subparser(
#         self, parser: argparse._SubParsersAction
#     ) -> argparse.ArgumentParser:
#         description = """Get tempurature scaled model"""
#         subparser = parser.add_parser(
#             self.name, description=description, help="Get tempurature scaled model."
#         )

#         subparser.add_argument(
#             "archive_file", type=str, help="path to an archived trained model"
#         )

#         subparser.add_argument(
#             "input_file",
#             type=str,
#             help="path to the file containing the evaluation data",
#         )

#         subparser.add_argument(
#             "--output-file",
#             type=str,
#             help="optional path to write the metrics to as JSON",
#         )

#         subparser.add_argument(
#             "--weights-file",
#             type=str,
#             help="a path that overrides which weights file to use",
#         )

#         cuda_device = subparser.add_mutually_exclusive_group(required=False)
#         cuda_device.add_argument(
#             "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
#         )

#         subparser.add_argument(
#             "-o",
#             "--overrides",
#             type=str,
#             default="",
#             help=(
#                 "a json(net) structure used to override the experiment configuration, e.g., "
#                 "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
#                 " with nested dictionaries or with dot syntax."
#             ),
#         )

#         subparser.add_argument(
#             "--batch-size",
#             type=int,
#             help="If non-empty, the batch size to use during evaluation.",
#         )

#         subparser.add_argument(
#             "--batch-weight-key",
#             type=str,
#             default="",
#             help="If non-empty, name of metric used to weight the loss on a per-batch basis.",
#         )

#         subparser.add_argument(
#             "--extend-vocab",
#             action="store_true",
#             default=False,
#             help="if specified, we will use the instances in your new dataset to "
#             "extend your vocabulary. If pretrained-file was used to initialize "
#             "embedding layers, you may also need to pass --embedding-sources-mapping.",
#         )

#         subparser.add_argument(
#             "--embedding-sources-mapping",
#             type=str,
#             default="",
#             help="a JSON dict defining mapping from embedding module path to embedding "
#             "pretrained-file used during training. If not passed, and embedding needs to be "
#             "extended, we will try to use the original file paths used during training. If "
#             "they are not available we will use random vectors for embedding extension.",
#         )
#         subparser.add_argument(
#             "--file-friendly-logging",
#             action="store_true",
#             default=False,
#             help="outputs tqdm status on separate lines and slows tqdm refresh rate",
#         )

#         subparser.set_defaults(func=temp_scale_from_args)

#         return subparser


# def temp_scale_from_args(args: argparse.Namespace) -> Dict[str, Any]:
#     common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging

#     # Disable some of the more verbose logging statements
#     logging.getLogger("allennlp.common.params").disabled = True
#     logging.getLogger("allennlp.nn.initializers").disabled = True
#     logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(
#         logging.INFO
#     )

#     # Load from archive
#     archive = load_archive(
#         args.archive_file,
#         weights_file=args.weights_file,
#         cuda_device=args.cuda_device,
#         overrides=args.overrides,
#     )
#     config = archive.config
#     prepare_environment(config)
#     model = archive.model
#     model.eval()

#     # Load the evaluation data
#     dataset_reader = archive.validation_dataset_reader

#     # output file
#     output_file_path = None
#     predictions_output_file_path = None

#     evaluation_data_path = args.input_file
#     if args.output_file != None:
#         output_file_path = args.output_file

#     logger.info("Reading evaluation data from %s", evaluation_data_path)
#     data_loader_params = config.get("validation_data_loader", None)
#     if data_loader_params is None:
#         data_loader_params = config.get("data_loader")
#     if args.batch_size:
#         data_loader_params["batch_size"] = args.batch_size
#     data_loader = DataLoader.from_params(
#         params=data_loader_params, reader=dataset_reader, data_path=evaluation_data_path
#     )

#     embedding_sources = (
#         json.loads(args.embedding_sources_mapping)
#         if args.embedding_sources_mapping
#         else {}
#     )

#     if args.extend_vocab:
#         logger.info("Vocabulary is being extended with test instances.")
#         model.vocab.extend_from_instances(instances=data_loader.iter_instances())
#         model.extend_embedder_vocab(embedding_sources)

#     data_loader.index_with(model.vocab)

#     temp_scaled_model = get_temp(
#         model, data_loader, args.cuda_device, output_file=output_file_path
#     )
#     logger.info("Finished calculating temp.")

#     return temp_scaled_model.temperature
