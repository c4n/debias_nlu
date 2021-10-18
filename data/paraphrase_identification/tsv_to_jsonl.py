import argparse
import json

import pandas as pd

NAME_MAPPING_TABLE = {
    "label": "is_duplicate"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv_file",
        required=True,
        help="Path to the original TSV file"
    )
    parser.add_argument(
        "--jsonl_file",
        required=True,
        help="Path to the destination jsonl file"
    )
    args = parser.parse_args()

    source_df = pd.read_csv(args.tsv_file, sep='\t')
    source_df.rename(columns=NAME_MAPPING_TABLE, inplace=True)

    f = open(args.jsonl_file, 'w')
    for datum in source_df.to_dict('records'):
        if pd.isna(datum["sentence1"]) or pd.isna(datum["sentence2"]):
            continue
        f.write("%s\n" % json.dumps(datum))

    f.close()
