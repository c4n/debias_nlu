#!/bin/bash

[ -d raw ] || mkdir raw

## Ref: https://github.com/TalSchuster/FeverSymmetric

wget -O fever.train.jsonl https://www.dropbox.com/s/v1a0depfg7jp90f/fever.train.jsonl
wget -O fever.dev.jsonl https://www.dropbox.com/s/bdwf46sa2gcuf6j/fever.dev.jsonl

wget -O fever_symmetric_v0.1.test.jsonl https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/symmetric_v0.1/fever_symmetric_generated.jsonl
wget -O fever_symmetric_v0.2.dev.jsonl https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/symmetric_v0.2/fever_symmetric_dev.jsonl
wget -O fever_symmetric_v0.2.test.jsonl https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/symmetric_v0.2/fever_symmetric_test.jsonl

## Ref: https://github.com/UKPLab/acl2020-confidence-regularization
wget -O raw/utama_bias_preds.json https://raw.githubusercontent.com/UKPLab/acl2020-confidence-regularization/master/biased_preds/weaker_balanced_duplicate_fever_claim_only.json
