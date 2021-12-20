import glob
import os
import fuse, causal_utils

import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.special import expit


### INIT ####
# select model
model_path='/raid/can/nli_models/baseline_mind_distill/nli/seed1/'
# select data path
data_path='/ist/users/canu/debias_nlu/data/nli/'
df = pd.read_json(model_path+'raw_train.jsonl', lines=True)
# select fusion method here
fusion = fuse.sum_fuse
#
df_hans = pd.read_json(data_path+'hans_prob_korn_lr_overlapping_sample_weight_3class.jsonl', lines=True)
hans_score=[b for b in df_hans['bias_probs'] ]
hans_score=np.array(hans_score)
list_probs = []
for i in df['probs']:
    list_probs.extend(i)
x=np.array(list_probs)
avg=np.average(x,axis=0)
bias_score=fusion(avg,hans_score)
y1m0=bias_score
result_path=model_path+'normal/'
# bert model predictions on HANS
df_bert = pd.read_json(result_path+'hans_result.jsonl', lines=True
# ent = []
y1m1prob = []
for p,h in zip(df_bert['probs'],hans_score):
    new_y1m1 = fusion(np.array(p),h)
    y1m1prob.append(new_y1m1)

##########
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image
import torch.nn as nn

class CustomImageDataset(Dataset):
    def __init__(self, probs,target_probs):
        self.probs  = torch.tensor(probs)
        self.target_probs = torch.tensor(target_probs)
    def __len__(self):
        return len(self.probs)

    def __getitem__(self, idx):
        return self.probs[idx], self.target_probs[idx]



class TempScale(nn.Module):
    def __init__(self):
        super(TempScale, self).__init__()
        self.T = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = (x/self.T)/torch.sum(x/self.T)
        return x
        
def train_loop(dataloader, model, loss_fn, optimizer):
    size = dataloader.__len__()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# init model
model = TempScale()

# hyperparam
learning_rate = 1e-3
batch_size = 64
epochs = 5
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#train
train=CustomImageDataset(df_bert['probs'],y1m1prob)
train_loop(train, model, loss_fn, optimizer)



