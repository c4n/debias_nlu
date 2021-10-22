from typing import List

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


LEMMATIZER = WordNetLemmatizer()


def vanilla_tokenize(_sent: str) -> List[str]:
    _words = [x.lower() for x in word_tokenize(_sent)]
    _words = list(filter(lambda x: x not in (".", "?", "!"), _words))
    return _words


def lemmatized_tokenize(_sent: str, _lemmatizer = LEMMATIZER) -> List[str]:
    _words = [_lemmatizer.lemmatize(x.lower(), "v") for x in word_tokenize(_sent)]
    _words = list(filter(lambda x: x not in (".", "?", "!"), _words))
    return _words