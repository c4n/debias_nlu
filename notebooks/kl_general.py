import os
from typing import List, Tuple, Union

import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.special import expit
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn

import fuse
import causal_utils


def sum_fuse(a, b):
    zsum = torch.add(a, b)
    zsum = torch.sigmoid(zsum)
    return torch.log(zsum)


### Configs ###
N_LABELS = 1
FUSE = sum_fuse

### INIT ####
# select model
model_path = '/raid/can/nli_models/baseline_mind_distill/nli/seed1/'
# select data path
data_path = '/raid/can/debias_nlu/data/nli/'
df = pd.read_json(model_path+'raw_train.jsonl', lines=True)
# select fusion method here
fusion = fuse.sum_fuse
#
print(data_path+'/raid/can/debias_nlu/data/nli/hans_prob_korn_lr_overlapping_sample_weight_3class.jsonl')
# exit()
df_hans = pd.read_json(
    data_path+'hans_prob_korn_lr_overlapping_sample_weight_3class.jsonl',
    lines=True
)
hans_score = [b for b in df_hans['bias_probs']]
hans_score = np.array(hans_score)
list_probs = []
for i in df['probs']:
    list_probs.extend(i)
x = np.array(list_probs)
avg = np.average(x, axis=0)
bias_score = fusion(avg, hans_score)
y1m0 = bias_score
result_path = model_path+'normal/'
# bert model predictions on HANS
df_bert = pd.read_json(result_path+'hans_result.jsonl', lines=True)
# ent = []
y1m1prob = []
for p, h in zip(df_bert['probs'], hans_score):
    new_y1m1 = fusion(np.array(p), h)
    y1m1prob.append(new_y1m1)


class CounterFactualDataset(Dataset):
    def __init__(self, probs, target_probs):
        self.probs = torch.tensor(probs)
        self.target_probs = torch.tensor(target_probs)

    def __len__(self):
        return len(self.probs)

    def __getitem__(self, idx):
        return self.probs[idx], self.target_probs[idx]


class CounterFactualModel(nn.Module):
    def __init__(self, n_labels: int = 1, init_c: Tuple[float] = None):
        super(CounterFactualModel, self).__init__()
        if init_c:
            assert n_labels == len(init_c)
            self.c = nn.Parameter(torch.tensor(init_c))
        else:
            const = 1.0 / float(n_labels)
            _init_c = const * np.ones(n_labels)
            self.c = nn.Parameter(torch.tensor(_init_c))

    def forward(self, x):
        x = FUSE(self.c, x)
        x = torch.nn.functional.softmax(x)
        return x


def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer) -> None:
    size = dataloader.__len__()
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y, model.c)
        loss.backward()
        optimizer.step()

        if batch % 2000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(
                f"batch: {batch},loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(model.c)


model = CounterFactualModel(n_labels=N_LABELS)

# hyper-params
learning_rate = 1e-3
batch_size = 64
epochs = 5
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def loss_fn(
    _pred: Union[List[float], List[List[float]]],
    _y: Union[List[float], List[List[float]]],
    _c: Union[List[float], List[List[float]]],
):
    cls_loss = torch.mean(
        -torch.multiply(
            _y,
            torch.log(_pred)
        )
    )
    kl_loss = torch.mean(
        -torch.multiply(
            _y,
            torch.log(_c)
        )
    )
    return cls_loss + kl_loss


# train
train = CounterFactualDataset(df_bert['probs'], y1m1prob)
train_loop(train, model, loss_fn, optimizer)
