import argparse
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

Y_COLUMN = "is_duplicate"
N_DEV_SET = 5000

NAME_MAPPING_TABLE = {
    "question1": "sentence1",
    "question2": "sentence2"
}

def _write_to_jsonl(_x_train: List[Dict], _y_train: List[Dict], path: str) -> None:
    f = open(path, 'w')
    for _x, _y in zip(_x_train, _y_train):
        f.write("%s\n" % {**_x, **_y})

    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv_file",
        default="raw/quora_duplicate_questions.tsv",
        help="Path to the original training TSV file"
    )
    parser.add_argument(
        "--dest_train",
        default="qqp.train.jsonl",
        help="Path to the destination training jsonl file"
    )    
    parser.add_argument(
        "--dest_val",
        default="qqp.dev.jsonl",
        help="Path to the destination validation jsonl file"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.tsv_file, sep='\t')
    df.rename(columns=NAME_MAPPING_TABLE, inplace=True)

    y = df.pop(Y_COLUMN).to_frame()
    X = df

    dev_size = float(N_DEV_SET) / float(len(df))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        stratify=y,
        test_size=dev_size,
        random_state=42
    )

    _write_to_jsonl(
        X_train.to_dict('records'),
        y_train.to_dict('records'),
        path=args.dest_train
    )


    _write_to_jsonl(
        X_val.to_dict('records'),
        y_val.to_dict('records'),
        path=args.dest_val
    )


