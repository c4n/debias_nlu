import argparse
import json
import math
from typing import Dict, List, Union

# Configs'
GROUND_TRUTH_LABEL = "gold_label"

# Ref: https://github.com/UKPLab/acl2020-confidence-regularization/blob/master/src/train_fever_distill.py
LABEL_MAP = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
REV_LABEL_MAP = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def _softmax(
    x: List[float]
) -> List[float]:
    x = list(map(lambda x: math.exp(x), x))
    divider = sum(x)
    return list(map(lambda x: x/divider, x))


def _merge(
    docs: List[Dict[str, Union[str, int]]],
    bias_preds: List[Dict[str, List[float]]],
    is_verbose: bool = 1
) -> List[Dict[str, Union[str, int, List[float]]]]:
    for doc in docs:
        index = str(doc["id"])
        try:
            distill_probs = _softmax(bias_preds[index])
        except KeyError:
            if is_verbose:
                print("Can not find id: %s from bias_preds." % index)
            continue
        grouth_truth_ans_idx = LABEL_MAP[doc[GROUND_TRUTH_LABEL]]
        doc["distil_probs"] = distill_probs
        doc["bias_prob"] = distill_probs[grouth_truth_ans_idx]
    return doc


def _read_jsonl(file_path: str) -> List[Dict[str, Union[str, int]]]:
    output = []
    f = open(file_path, 'r')
    line = f.readline()
    while line:
        doc = json.loads(line)
        output.append(doc)
        line = f.readline()
    f.close()
    return output


def _write_jsonl(docs: List[Dict[str, Union[str, int, List[float]]]], path: str) -> None:
    f = open(path, 'w')
    for doc in docs:
        f.write("%s\n" % json.dumps(doc))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--org_train_jsonl",
    )
    parser.add_argument(
        "--utama_json",
    )
    parser.add_argument(
        "--output_jsonl",
    )
    parser.add_argument(
        "--is_verbose", default=1  # allow  0 or 1
    )
    args = parser.parse_args()

    is_verbose = int(args.is_verbose)

    docs = _read_jsonl(args.org_train_jsonl)
    bias_preds = json.load(open(args.utama_json, "r"))
    merged_data = _merge(
        docs=docs,
        bias_preds=bias_preds,
        is_verbose=is_verbose
    )

    _write_jsonl(
        docs=docs,
        path=args.output_jsonl
    )
