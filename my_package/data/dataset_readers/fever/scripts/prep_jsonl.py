import argparse
import json
import os
from tqdm import tqdm

import sys
sys.path.append("../../")
from fever.wikis import ESWikiLookUp
from fever.datamodels import EvidenceList


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--source', type=str, default="../tests/test-raw-data.jsonl",
        help='Path for raw .jsonl file'
    )
    parser.add_argument(
        '--target', type=str, default="../tests/test-duplicate.jsonl",
        help='Target output file of the combined text wiki in jsonl format'
    )
    args = parser.parse_args()

    source = args.source
    target = args.target
    

    n_docs = sum(1 for i in open(source, 'rb'))
    wklp = ESWikiLookUp()

    ft = open(target, 'w')
    ft.truncate()

    with tqdm(total=n_docs) as pbar:
        with open(source, 'r') as fs:
            line = fs.readline()
            while line:
                doc = json.loads(line)
                # print("claim: ", doc["claim"])
                # print("evidence: ", doc["evidence"])
                evidences = EvidenceList.from_list(
                    evs=doc["evidence"], wklp=wklp
                )
                doc["evidence"] = evidences.to_text()
                ft.write("%s\n"%doc)

                pbar.update(1)
                line = fs.readline()

    ft.close()
