#!/bin/bash

## https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs
wget -O raw/quora_duplicate_questions.tsv http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv

## https://github.com/google-research-datasets/paws
wget -O raw/paws_qqp.tar.gz https://storage.googleapis.com/paws/english/paws_qqp.tar.gz
tar -xvf raw/paws_qqp.tar.gz -C raw/
