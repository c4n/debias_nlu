#!/bin/bash

## Ref: https://github.com/google-research-datasets/paws

ORIGINAL_QQP_FILE=${1:-raw/quora_duplicate_questions.tsv}
PAWS_QQP_DIR=${2:-raw/paws_qqp}
PAWS_QQP_OUTPUT_DIR=${2:-paws_qqp}

[ -d $PAWS_QQP_DIR ] || mkdir $PAWS_QQP_DIR

## QQP

python split_qqp_train_val.py --tsv_file $ORIGINAL_QQP_FILE

## PAWS

python qqp_generate_data.py \
  --original_qqp_input="${ORIGINAL_QQP_FILE}" \
  --paws_input="${PAWS_QQP_DIR}/train.tsv" \
  --paws_output="${PAWS_QQP_OUTPUT_DIR}/train.tsv"

python qqp_generate_data.py \
  --original_qqp_input="${ORIGINAL_QQP_FILE}" \
  --paws_input="${PAWS_QQP_DIR}/dev_and_test.tsv" \
  --paws_output="${PAWS_QQP_OUTPUT_DIR}/dev_and_test.tsv"

python tsv_to_jsonl.py \
  --tsv_file="${PAWS_QQP_OUTPUT_DIR}/train.tsv" \
  --jsonl_file="paws.train.jsonl"

python tsv_to_jsonl.py \
  --tsv_file="${PAWS_QQP_OUTPUT_DIR}/dev_and_test.tsv" \
  --jsonl_file="paws.dev_and_test.jsonl"
