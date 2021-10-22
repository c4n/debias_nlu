from typing import Dict, List

from math import log


def lmi(
    p_w_l: float,
    p_l_given_w: float,
    p_l: float
) -> float:
    return p_w_l * log(p_l_given_w/p_l)


def get_ngram_probs(
    ngram_docs: List[List[str]],
    labels: List[str],
    possible_labels: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
        Example of the output:
            {
                "SUPPORTS": {
                    "does_not": {
                        "p_w_l": 0.001,
                        "p_l_given_w": 0.005,
                        "p_l": 0.002
                    },
                    "get_in": {
                        "p_w_l": 0.001,
                        "p_l_given_w": 0.005,
                        "p_l": 0.002
                    }
                },
                "REFUTES": {
                    "did_not": {
                        "p_w_l": 0.001,
                        "p_l_given_w": 0.005,
                        "p_l": 0.002
                    }
                }
            }
    """
    possible_labels = possible_labels if possible_labels else list(set(labels))
    counter = {label: {} for label in possible_labels} # count(w, l)
    n_appear_labels = {label: 0 for label in possible_labels} # count(l)
    n_ngrams = {} # count(w)

    for ngram_doc, label in zip(ngram_docs, labels):
        for ngram in ngram_doc:
            counter[label][ngram] = counter[label].get(ngram, 0) + 1
            n_ngrams[ngram] = n_ngrams.get(ngram, 0) + 1
            n_appear_labels[label] += 1

    total_ngrams = sum([n for _, n in n_appear_labels.items()]) # D
    prob = {label: {} for label in possible_labels}

    for label in possible_labels:
        p_l = n_appear_labels[label] / total_ngrams
        for ngram in counter[label].keys():
            prob[label][ngram] = {
                "p_w_l": counter[label][ngram] / total_ngrams,
                "p_l_given_w": counter[label][ngram] / n_ngrams[ngram],
                "p_l": p_l
            }

    return prob


def compute_lmi(
    ngram_docs: List[List[str]],
    labels: List[str],
    possible_labels: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
        Example of the output:
            {
                "SUPPORTS": {
                    "does_not": 0.2,
                    "get_in": 0.1
                },
                "REFUTES": {
                    "did_not": 0.2
                }
            }
    """
    possible_labels = possible_labels if possible_labels else list(set(labels))
    ngram_probs = get_ngram_probs(
        ngram_docs=ngram_docs,
        labels=labels,
        possible_labels=possible_labels
    )
    for label in possible_labels:
        for ngrams in ngram_probs[label].keys():
            ngram_probs[label][ngrams] = lmi(**ngram_probs[label][ngrams])
    return ngram_probs