import argparse
import copy
import json
from typing import Dict, List, Union


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
    parser.add_argument("--in_jsonl")
    parser.add_argument("--out_jsonl")
    parser.add_argument("--old_key")
    parser.add_argument("--new_key")
    args = parser.parse_args()

    org_jsons = _read_jsonl(args.in_jsonl)

    final_jsons = []
    for item in org_jsons:
        _item = copy.deepcopy(item)
        _item[args.new_key] = _item[args.old_key]
        del _item[args.old_key]
        final_jsons.append(_item)

    _write_jsonl(
        docs=final_jsons,
        path=args.out_jsonl
    )
