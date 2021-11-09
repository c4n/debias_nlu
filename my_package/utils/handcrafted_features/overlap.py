from typing import List, Union

import spacy

from ..tokenizer import vanilla_tokenize


SPACY_NLP = spacy.load("en_core_web_sm")


def get_lexical_overlap(
    sent1: Union[str, List[str]],
    sent2: Union[str, List[str]]
) -> float:
    sent1 = sent1 if isinstance(sent1, list) else vanilla_tokenize(sent1)
    sent2 = sent2 if isinstance(sent2, list) else vanilla_tokenize(sent2)

    count = 0
    for w1 in sent1:
        for w2 in sent2:
            if w1 == w2:
                count += 1

    return count / max(len(sent1), len(sent2))


def get_entities_overlap(
    sent1: str,
    sent2: str
) -> int:
    doc1 = SPACY_NLP(sent1)
    doc2 = SPACY_NLP(sent2)

    count = 0
    for ent1 in doc1.ents:
        for ent2 in doc2.ents:
            if (ent1.text, ent1.label_) == (ent2.text, ent2.label_):
                count += 1

    return count
