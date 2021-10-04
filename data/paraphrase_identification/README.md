## Paraphase identification

The original data from QQP and the challenge set is PAWS.

## Steps to get the preprocessed data

1. download data as in the command

```bash
bash download.sh
```

2. preprocess QQP on the local to paws since there is a copy right issue by running

```bash
bash preprocess.sh
```
Please note that this step requires NLTK on your python env

3. The expected outputs would be 3 jsonl files in "data/paraphrase_identification" path which are
```
1. qqp.train.jsonl
2. qqp.dev.jsonl
3. paws.dev_and_test.jsonl
```