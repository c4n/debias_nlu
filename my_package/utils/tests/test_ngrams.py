from unittest import TestCase

from ..ngrams import get_ngram_docs
from ..tokenizer import vanilla_tokenize


class TestNGrams(TestCase):
    def test_unigram(self):
        docs = ["I love you.", "Hello there. Let me run."]
        n = 1

        ngram_docs = get_ngram_docs(docs, n=n, tokenize=vanilla_tokenize)
        self.assertListEqual(ngram_docs[0], ["i", "love", "you"])
        self.assertListEqual(ngram_docs[1], ["hello", "there", "let", "me", "run"])

    def test_bigram(self):
        docs = ["I love you.", "Hello there. Let me run."]
        n = 2

        ngram_docs = get_ngram_docs(docs, n=n, tokenize=vanilla_tokenize)
        self.assertListEqual(ngram_docs[0], ["i_love", "love_you"])
        self.assertListEqual(ngram_docs[1], ["hello_there", "there_let", "let_me", "me_run"])