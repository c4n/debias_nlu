import argparse
import json
from typing import Dict, List, Union

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_Y_COLUMN = "gold_label"
DEFAULT_N_DEV_SET = 5000
FIXED_RANDOM_SEED = 42


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


def _write_jsonl(_x_train: List[Dict], _y_train: List[Dict], path: str) -> None:
    f = open(path, 'w')
    for _x, _y in zip(_x_train, _y_train):
        f.write("%s\n" % json.dumps({**_x, **_y}))

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original",
    )
    parser.add_argument(
        "--dest_train",
    )
    parser.add_argument(
        "--dest_val",
        help="Path to the destination validation jsonl file"
    )
    parser.add_argument(
        "--n_dev_set",
        default=DEFAULT_N_DEV_SET
    )
    parser.add_argument(
        "--y_column",
        default=DEFAULT_Y_COLUMN
    )
    parser.add_argument(
        "--is_stratify",
        default="1"
    )
    args = parser.parse_args()

    n_dev_set = int(args.n_dev_set)
    is_stratify = True if args.is_stratify in (1, "1", "True", "true") \
        else False
    y_col = args.y_column

    data = _read_jsonl(args.original)
    df = pd.DataFrame(data)

    y = df.pop(y_col).to_frame()
    X = df

    dev_size = float(n_dev_set) / float(len(df))
    if is_stratify:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            stratify=y,
            test_size=dev_size,
            random_state=FIXED_RANDOM_SEED
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=dev_size,
            random_state=FIXED_RANDOM_SEED
        )

    _write_jsonl(
        X_train.to_dict('records'),
        y_train.to_dict('records'),
        path=args.dest_train
    )

    _write_jsonl(
        X_val.to_dict('records'),
        y_val.to_dict('records'),
        path=args.dest_val
    )
