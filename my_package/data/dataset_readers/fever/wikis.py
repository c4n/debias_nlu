from abc import ABC, abstractmethod
import logging
import json
from multiprocessing import Pool
import os
import requests
from typing import List, Tuple
import unicodedata

class WikiLookUp(ABC):
    @abstractmethod
    def find_text_by_url(self, url: str) -> str:
        ...


DEFAULT_STATIC_WIKI_PATH = "./wiki-pages"

DEFAULT_ES_HOST = "http://localhost:9200"
DEFAULT_ES_INDEX = "wiki"


class StaticWikiLookUp(WikiLookUp):
    def __init__(
        self,
        path: str = DEFAULT_STATIC_WIKI_PATH,
        n_workers: int = None
    ) -> None:
        self.path = path
        self.n_workers = n_workers

    def find_text_by_url(self, url: str) -> str:
        files = os.listdir(self.path)
        if self.n_workers:
            pool = Pool(processes=self.n_workers)
            results = pool.map(
                StaticWikiLookUp.match_in_jsonl,
                [(url, os.path.join(self.path, f)) for f in files]
            )
            results = list(filter(lambda x: x != {}, results))
            return None if len(results) == 0 else results[0]["lines"]
        for f in files:
            result = StaticWikiLookUp.match_in_jsonl(
                (url, os.path.join(self.path, f)))
            if result != {}:
                return result["lines"]
        return None

    @staticmethod
    def match_in_jsonl(arg: Tuple[str]) -> dict:
        url = arg[0]
        file = arg[1]
        with open(file, 'r') as fh:
            line = fh.readline()
            while line:
                doc = json.loads(line)
                if url == doc["id"]:
                    return doc
                line = fh.readline()
        return {}


class ESWikiLookUp(WikiLookUp):
    LOGGER = logging.getLogger()

    def __init__(
        self,
        host: str = DEFAULT_ES_HOST,
        index: str = DEFAULT_ES_INDEX
    ) -> None:
        self.host = host
        self.index = index

    def _query(self, url: str, is_sensitive: bool = True) -> dict:
        if is_sensitive:
            data = json.dumps({
                "query": {
                    "match_phrase": {
                        "id": url
                    }
                }
            })
        else:
            data = json.dumps({
                "query": {
                    "match" : {
                        "id": {
                            "query": url,
                            "fuzziness": "2"
                        }
                    }
                }
            })

        response = requests.get(
            "%s/%s/_search"%(self.host, self.index),
            headers = {'Content-type': 'application/json'},
            data = data
        )
        if response.status_code != 200:
            raise SystemError("%d: %s"%(response.status_code, response.content))
        return response.json()["hits"]["hits"]

    def find_text_by_url(self, url: str) -> str:
        results = self._query(url=url)
        if results:
            for result in results:
                if url == result["_source"]["id"]:
                    return result["_source"]["lines"]
        # French char sim case
        new_results = self._query(url=url, is_sensitive=False)
        if new_results:
            for result in new_results:
                if unicodedata.normalize('NFKD', url) == unicodedata.normalize('NFKD', result["_source"]["id"]):
                    return result["_source"]["lines"]

        ESWikiLookUp.LOGGER.warning("Search URL: '%s' can not be found!"%url)
        return None