#!/bin/bash

python split_train_val.py \
    --original fever.train.jsonl \
    --dest_train fever.train.jsonl \
    --dest_val fever.val.jsonl \
    --n_dev_set 5000 \
    --is_stratify 1

python merge_utama_probs.py \
    --org_train_jsonl fever.train.jsonl \
    --utama_json raw/utama_bias_preds.json \
    --output_jsonl utama_fever.train.jsonl \
    --is_verbose 1