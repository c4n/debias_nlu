import re
from typing import List

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


LEMMATIZER = WordNetLemmatizer()


def __filter(_sent: List[str]) -> List[str]:
    _sent = list(filter(lambda x: re.match("\w+", x), _sent))
    return _sent


def vanilla_tokenize(_sent: str, _filter=__filter) -> List[str]:
    _words = [x.lower() for x in word_tokenize(_sent)]
    _words = _filter(_words)
    return _words


def lemmatized_tokenize(_sent: str, _lemmatizer=LEMMATIZER, _filter=__filter) -> List[str]:
    _words = [_lemmatizer.lemmatize(x.lower(), "v")
              for x in word_tokenize(_sent)]
    _words = _filter(_words)
    return _words
