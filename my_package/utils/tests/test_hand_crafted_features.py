import math
from unittest import TestCase

from ..ngrams import get_ngram_docs
from ..tokenizer import vanilla_tokenize
from ..handcrafted_features.counter import count_negations
from ..handcrafted_features.mutual_information import compute_lmi
from ..handcrafted_features.overlap import get_entities_overlap, get_lexical_overlap


class TestCounter(TestCase):
    def test_negation_counter_case_no_negation(self):
        sent = "I did attend the university."
        n_negations = count_negations(sent)
        self.assertEqual(n_negations, 0)

    def test_negation_counter_case_one_negation(self):
        sent = "I didn't attend the university."
        n_negations = count_negations(sent)
        self.assertEqual(n_negations, 1)

    def test_negation_counter_case_multiple_negations_one_sentence(self):
        sent = "I didn't attend the university because I do not enjoy studying."
        n_negations = count_negations(sent)
        self.assertEqual(n_negations, 2)

    def test_negation_counter_case_multiple_negations_multiple_sentences(self):
        sent = "I didn't attend the university. I do not enjoy studying."
        n_negations = count_negations(sent)
        self.assertEqual(n_negations, 2)


class TestEntityOverlap(TestCase):
    def test_no_entity_overlap(self):
        sent1 = "I have no clue."
        sent2 = "I also have no clue. I also go to the college."
        n_overlap = get_entities_overlap(sent1=sent1, sent2=sent2)
        self.assertEqual(n_overlap, 0)

    def test_one_entity_overlap_1(self):
        sent1 = "I was in Hawaii."
        sent2 = "When I was in Hawaii, I went to the beach."
        n_overlap = get_entities_overlap(sent1=sent1, sent2=sent2)
        self.assertEqual(n_overlap, 1)

    def test_one_entity_overlap_2(self):
        sent1 = "I was in Hawaii and Chicago."
        sent2 = "When I was in Hawaii, I went to the beach. By the way, I was born in New York."
        n_overlap = get_entities_overlap(sent1=sent1, sent2=sent2)
        self.assertEqual(n_overlap, 1)

    def test_two_entity_overlap(self):
        sent1 = "I was in Hawaii and Chicago."
        sent2 = "When I was in Hawaii, I went to the beach. By the way, I was born in Chicago."
        n_overlap = get_entities_overlap(sent1=sent1, sent2=sent2)
        self.assertEqual(n_overlap, 2)


class TestMutualInformation(TestCase):
    def test_compute_lmi_bigram(self):
        docs = [
            "I love you.",
            "We love you.",
            "They hate you."
        ]
        n = 2
        labels = ["SUPPORTS", "SUPPORTS", "REFUFUTES"]

        ngram_docs = get_ngram_docs(docs=docs, n=n, tokenize=vanilla_tokenize)
        conmputed_lmi = compute_lmi(ngram_docs=ngram_docs, labels=labels)

        self.assertAlmostEqual(
            conmputed_lmi["SUPPORTS"]["i_love"],
            (1/6) * math.log((1/1)/(4/6)),
            places=4
        )
        self.assertAlmostEqual(
            conmputed_lmi["SUPPORTS"]["love_you"],
            (2/6) * math.log((2/2)/(4/6)),
            places=4
        )
        self.assertAlmostEqual(
            conmputed_lmi["REFUFUTES"]["they_hate"],
            (1/6) * math.log((1/1)/(2/6)),
            places=4
        )
        


class TestLexicalOverlap(TestCase):
    def test_no_lexical_overlap(self):
        sent1 = "I was in Hawaii and Chicago."
        sent2 = "We went there."
        overlap_score = get_lexical_overlap(sent1=sent1, sent2=sent2)
        self.assertAlmostEqual(overlap_score, 0, 3)

    def test_half_lexical_overlap(self):
        sent1 = "I eat."
        sent2 = "I run."
        overlap_score = get_lexical_overlap(sent1=sent1, sent2=sent2)
        self.assertAlmostEqual(overlap_score, 0.5, 3)

    def test_a_quater_of_lexical_overlap(self):
        sent1 = "I eat. She swim."
        sent2 = "I run. They gone."
        overlap_score = get_lexical_overlap(sent1=sent1, sent2=sent2)
        self.assertAlmostEqual(overlap_score, 0.25, 3)