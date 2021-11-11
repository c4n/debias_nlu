#!/bin/bash

wget -O fever.train.jsonl https://www.dropbox.com/s/v1a0depfg7jp90f/fever.train.jsonl
wget -O fever.dev.jsonl https://www.dropbox.com/s/bdwf46sa2gcuf6j/fever.dev.jsonl

## Ref: https://github.com/TalSchuster/FeverSymmetric
wget -O fever_symmetric_v0.1.jsonl https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/symmetric_v0.1/fever_symmetric_generated.jsonl
wget -O fever_symmetric_v0.2.jsonl https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/symmetric_v0.2/fever_symmetric_test.jsonl