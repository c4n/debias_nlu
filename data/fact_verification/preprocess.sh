#!/bin/bash

python split_train_val.py \
    --original fever.train.jsonl \
    --dest_train fever.train.jsonl \
    --dest_val fever.val.jsonl \
    --n_dev_set 5000 \
    --is_stratify 1