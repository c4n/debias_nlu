import os
import random
from typing import List, Tuple, Union

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn

import fuse


MY_RANDOM_SEED = int(os.getenv("MY_RANDOM_SEED", 42))
torch.manual_seed(MY_RANDOM_SEED)
TORCH_GENERATOR = torch.Generator()
TORCH_GENERATOR.manual_seed(MY_RANDOM_SEED)
random.seed(MY_RANDOM_SEED)
np.random.seed(MY_RANDOM_SEED)


def torch_sum_fuse(a: torch.Tensor, b: torch.Tensor):
    zsum = torch.add(a, b)
    zsum = torch.sigmoid(zsum)
    return torch.log(zsum)


def torch_mult_fuse(a: torch.Tensor, b: torch.Tensor):
    smax = torch.nn.Softmax(dim=1)
    return smax(torch.mul(a, b))


### Configs ###
DEFAULT_CONFIG = {
    "N_LABELS": 3,
    "FUSE": torch_sum_fuse,

    "EPOCHS": 16,
    "BATCH_SIZE": 64,
    "LEARNING_RATE": 0.0001
}

TE_CONFIG = {
    "N_LABELS": 3,
    "FUSE": torch_mult_fuse,

    "EPOCHS": 16,
    "BATCH_SIZE": 64,
    "LEARNING_RATE": 0.0001
}


class CounterFactualDataset(Dataset):
    def __init__(self, probs, target_probs):
        self.probs = torch.tensor(probs)
        self.target_probs = torch.tensor(target_probs)

    def __len__(self):
        return len(self.probs)

    def __getitem__(self, idx):
        return self.probs[idx], self.target_probs[idx]


class CounterFactualModel(nn.Module):
    def __init__(self, n_labels: int = 1, init_c: Tuple[float] = None, fuse=DEFAULT_CONFIG["FUSE"]):
        super(CounterFactualModel, self).__init__()
        self.fuse = fuse
        self.n_labels = n_labels
        if init_c:
            assert n_labels == len(init_c)
            self.c = nn.Parameter(torch.tensor(init_c))
        else:
            const = 1.0 / float(n_labels)
            _init_c = const  # * np.ones(n_labels)
            self.c = nn.Parameter(torch.tensor(_init_c))

    def forward(self, x):
        temp_ones = torch.ones(self.n_labels).detach()
        temp_c = self.c * temp_ones
        x = self.fuse(temp_c, x)
        return x


def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn,
    optimizer,
    scheduler=None,
    verbose: bool = False
) -> None:
    size = dataloader.__len__()
    for batch, (bert_probs, bias_model_logits) in enumerate(dataloader):
        optimizer.zero_grad()
        # Compute prediction and loss
        pred = model(bias_model_logits)
        loss = loss_fn(
            _bert_pred=bert_probs,
            _masked_pred=torch.nn.functional.softmax(pred, dim=1)
        )
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
    if verbose:
        print(f"loss: {loss:>7f}")
        print("c: ", model.c)
        print("softmax(c): ", torch.nn.functional.softmax(model.c, dim=0))


def loss_fn(
    _bert_pred: Union[List[float], List[List[float]]],
    _masked_pred: Union[List[float], List[List[float]]]
):
    # Ref from CF-VQA
    # kl_loss = torch.mean(
    #     -torch.multiply(
    #         _bert_pred,
    #         torch.log(_masked_pred)
    #     ),
    #     dim=1
    # )
    # return torch.mean(kl_loss)
    new_kl_loss = torch.mean(
        torch.multiply(
            _bert_pred,
            torch.log(
                torch.div(
                    _bert_pred,
                    _masked_pred
                )
            )
        ),
        dim=1
    )
    return torch.mean(new_kl_loss)


def sharpness_correction(
    bert_pred_probs: List[List[float]],
    y1m1probs: List[List[float]],
    verbose: bool = False,
    config: dict = DEFAULT_CONFIG
) -> List[float]:
    if verbose:
        print("Config: ", config)

    model = CounterFactualModel(n_labels=config["N_LABELS"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["LEARNING_RATE"]
    )

    # train
    dataset = CounterFactualDataset(bert_pred_probs, y1m1probs)
    dataloader = DataLoader(
        dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        generator=TORCH_GENERATOR
    )
    for ep in range(config["EPOCHS"]):
        if verbose:
            print("======== EPOCH %d ========" % (ep+1))
        train_loop(dataloader, model, loss_fn, optimizer, verbose=verbose)

    return model.c.detach().cpu().numpy()


if __name__ == "__main__":
    '''
        This is just for a test
    '''
    ### INIT ####
    # select model
    model_path = '/raid/can/nli_models/baseline_mind_distill/nli/seed1/'
    # select data path
    data_path = '/raid/can/debias_nlu/data/nli/'
    df = pd.read_json(model_path+'raw_train.jsonl', lines=True)
    # select fusion method here
    fusion = fuse.sum_fuse
    #
    df_hans = pd.read_json(
        data_path+'dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl', lines=True)
    hans_score = [b for b in df_hans['bias_probs']]
    hans_score = np.array(hans_score)
    list_probs = []
    for i in df['probs']:
        list_probs.extend(i)
    x = np.array(list_probs)
    avg = np.average(x, axis=0)
    bias_score = fusion(avg, hans_score)
    y1m0 = bias_score
    result_path = model_path
    # bert model predictions on HANS
    df_bert = pd.read_json(result_path+'raw_m.jsonl', lines=True)
    y1m1prob = []
    for p, h in zip(df_bert['probs'], hans_score):
        new_y1m1 = fusion(np.array(p), h)
        y1m1prob.append(new_y1m1)

    output = sharpness_correction(
        bert_pred_probs=df_bert['probs'],
        y1m1probs=y1m1prob,
        verbose=True
    )

    print("output: ", output, type(output))
