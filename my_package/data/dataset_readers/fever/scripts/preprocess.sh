#!/bin/bash

# ./start_es.sh

until curl -s -f -o /dev/null "http://localhost:9200"
do
  sleep 2
done

WIKIPATH=${1:-../../wiki-pages/}

TRAIN_SOURCE=${2:-../../data/train.jsonl}
TRAIN_TARGET=${3:-../../data/prep/train_duplicate.jsonl}
TEST_SOURCE=${4:-../../data/paper_test.jsonl}
TEST_TARGET=${5:-../../data/prep/paper_test_duplicate.jsonl}
DEV_SOURCE=${6:-../../data/paper_dev.jsonl}
DEV_TARGET=${7:-../../data/prep/paper_dev_duplicate.jsonl}

### Wiki to ElasticSearch
# for f in $(ls $WIKIPATH); do
#   elasticsearch_loader --index wiki --type incident json --json-lines "$WIKIPATH/$f"
# done
# elasticsearch_loader --index wiki --type incident json --json-lines ../tests/test-wiki.jsonl

### Preprocessing to new jsonl files
python prep_jsonl.py --source $TRAIN_SOURCE --target $TRAIN_TARGET
python prep_jsonl.py --source $TEST_SOURCE --target $TEST_TARGET
python prep_jsonl.py --source $DEV_SOURCE --target $DEV_TARGET
# python prep_jsonl.py --source ../tests/test-raw-data.jsonl --target ../tests/test-duplicate.jsonl

# ./stop_es.sh