import argparse
import json
import requests
import logging
import os
from typing import Tuple

from tqdm import tqdm

INDEX = "wiki"
SELECTED_ATTRS = ("id", "text")

def create_index(host: str, port: int, index: str = INDEX) -> None:
    url = 'http://%s:%d/%s'%(host, port, index)
    if requests.get(url).status_code == 200:
        response = requests.delete(url)
        if response.status_code != 200:
            raise requests.RequestException(response.json())

    response = requests.put(url)
    if response.status_code != 200:
        raise IndexError(response.content.json())

def insert(host: str, port: int, doc: dict, index: str = INDEX, selected_attrs: Tuple[str] = SELECTED_ATTRS) -> None:
    response = requests.post(
        'http://%s:%d/%s/_doc'%(host, port, index),
        json = {k:v for k,v in doc.items() if k in selected_attrs},
        headers = {'Content-type': 'application/json'}
    )
    if response.status_code != 201:
        raise requests.RequestException(response.json())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--path', type=str,
        help='Path for all wiki-pages in json lines'
    )
    parser.add_argument(
        '--host', type=str, default="localhost",
        help='Elasticsearch host'
    )
    parser.add_argument(
        '--port', type=int, default=9200,
        help='Elasticsearch REST port'
    )
    args = parser.parse_args()

    host = args.host
    port = args.port
    wiki_path = args.path

    create_index(host=host, port=port)

    for f in tqdm(os.listdir(wiki_path)):
        with open(os.path.join(wiki_path, f), 'r') as fh:
            line = fh.readline()
            while line:
                doc = json.loads(line)
                insert(
                    host=host, port=port,
                    doc=doc
                )
                line = fh.readline()                