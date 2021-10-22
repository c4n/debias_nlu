from typing import List, Callable

from .tokenizer import vanilla_tokenize


def get_ngram_doc(
    doc: str,
    n: int,
    tokenize: Callable[[str], List[str]] = vanilla_tokenize
)-> List[str]:
    tokenized_doc = tokenize(doc)
    length = len(tokenized_doc)-(n-1)
    return [
        '_'.join([tokenized_doc[i+j] for j in range(n)])
        for i in range(length)
    ]


def get_ngram_docs(
    docs: List[str],
    n: int,    
    tokenize: Callable[[str], List[str]] = vanilla_tokenize
) -> List[List[str]]:
    return [
        get_ngram_doc(doc=doc, n=n, tokenize=tokenize)
        for doc in docs
    ]
