import os
import operator
import pickle
from typing import Callable, Dict, List, Tuple

from sklearn.base import RegressorMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from my_package.models.traditional import TraditionalML, FEATURE_EXTRACTOR
from my_package.utils.handcrafted_features.mutual_information import compute_lmi
from my_package.utils.ngrams import get_ngram_doc, get_ngram_docs
from my_package.utils.tokenizer import vanilla_tokenize


DEFAULT_CONFIG = {
    "n_grams": [1, 2],
    "top_ks": [50, 50],
}

DEFAULT_MODEL = LogisticRegression(
    random_state=42,
    solver='saga',
    max_iter=500
)


class Classifier(TraditionalML):
    def __init__(
        self,
        possible_labels: List[str],
        feature_extractors: List[FEATURE_EXTRACTOR],
        tokenizer: Callable[[str], List[str]] = vanilla_tokenize,
        normalizer: TransformerMixin = MinMaxScaler(),
        model: RegressorMixin = DEFAULT_MODEL,
        config: dict = DEFAULT_CONFIG
    ) -> None:
        """
            Currently, tokenizer and normalizer can not be saved to the file.
            We have to inject it mannually.
        """
        self.map_labels = {lb: i for i, lb in enumerate(possible_labels)}
        self.model = model
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.feature_extractors = feature_extractors

        self.config = config
        self._validate_config()
        # internal states
        self.top_ngrams_sent1 = None
        self.top_ngrams_sent2 = None
        self.words_to_idx = None
        self.n_features = None

    def save(self, folder: str) -> None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        _model = {
            "map_labels": self.map_labels,
            "model": self.model,
            "normalizer": self.normalizer,
        }
        _states = {
            "config": self.config,
            "top_ngrams_sent1": self.top_ngrams_sent1,
            "top_ngrams_sent2": self.top_ngrams_sent2,
            "words_to_idx": self.words_to_idx,
            "n_features": self.n_features
        }
        pickle.dump(_model, open(os.path.join(folder, "model.pickle"), 'wb'))
        pickle.dump(_states, open(os.path.join(folder, "state.pickle"), 'wb'))

    def load(self, folder: str) -> None:
        _model = pickle.load(open(os.path.join(folder, "model.pickle"), 'rb'))
        self.map_labels = _model["map_labels"]
        self.model = _model["model"]
        self.normalizer = _model["normalizer"]
        _states = pickle.load(open(os.path.join(folder, "state.pickle"), 'rb'))
        self.config = _states["config"]
        self.top_ngrams_sent1 = _states["top_ngrams_sent1"]
        self.top_ngrams_sent2 = _states["top_ngrams_sent2"]
        self.words_to_idx = _states["words_to_idx"]
        self.n_features = _states["n_features"]

    def _validate_config(self):
        assert len(self.config.get("n_grams")) \
            == len(self.config.get("top_ks"))

    def _get_top_n_grams(self, docs: List[str], labels: List[str]) -> Dict[int, List[str]]:
        top_ngrams = {}
        for n, top_k in zip(self.config.get("n_grams"), self.config.get("top_ks")):
            ngram_docs = get_ngram_docs(
                docs=docs, n=n,
                tokenize=self.tokenizer
            )
            lmis = compute_lmi(ngram_docs=ngram_docs, labels=labels)
            top_k_lmis = {
                label: dict(sorted(
                    lmi.items(), key=operator.itemgetter(1), reverse=True
                )[:top_k])
                for label, lmi in lmis.items()
            }
            if self.config.get("verbose", False):
                print("%d-gram LMI: " % n, top_k_lmis, "\n")
            top_ngrams[n] = []
            for _, lmi in top_k_lmis.items():
                top_ngrams[n].extend(lmi.keys())
        return top_ngrams

    def _transform(self, doc: Tuple[str, str]) -> List[float]:
        n_tokens = len(self.map_labels) * sum(self.config.get("top_ks"))
        vec_output = [0, ] * self.n_features

        for n in self.config.get("n_grams"):
            ngram_sent1 = get_ngram_doc(
                doc[0], n=n,
                tokenize=self.tokenizer
            )
            for i, token in enumerate(ngram_sent1):
                if token in self.words_to_idx:
                    idx = self.words_to_idx[token]
                    vec_output[idx] += 1

            ngram_sent2 = get_ngram_doc(
                doc[1], n=n,
                tokenize=self.tokenizer
            )
            for i, token in enumerate(ngram_sent2):
                if token in self.words_to_idx:
                    idx = self.words_to_idx[token]
                    vec_output[idx] += 1

        for i, f in enumerate(self.feature_extractors):
            idx = 2 * n_tokens + i
            vec_output[idx] = f(doc[0], doc[1])
        return vec_output

    def fit(self, docs: List[Tuple[str, str]], labels: List[str]) -> None:
        sent1s = [d[0] for d in docs]
        sent2s = [d[1] for d in docs]
        if self.config.get("verbose", False):
            print("------ Top N-grams for sentence 1 ------")
        self.top_ngrams_sent1 = self._get_top_n_grams(sent1s, labels)
        if self.config.get("verbose", False):
            print("------ Top N-grams for sentence 2 ------")
        self.top_ngrams_sent2 = self._get_top_n_grams(sent2s, labels)

        self.words_to_idx = {}
        for top_ngrams_sent_i in (self.top_ngrams_sent1,  self.top_ngrams_sent2):
            for n in top_ngrams_sent_i:
                for w in top_ngrams_sent_i[n]:
                    self.words_to_idx[w] = len(self.words_to_idx)

        self.n_features = 2 * \
            len(self.map_labels) * sum(self.config.get("top_ks")) + \
            len(self.feature_extractors)
        if self.config.get("verbose", False):
            print("n_features: %d" % self.n_features)

        x = [self._transform(d) for d in docs]
        y = [self.map_labels[lb] for lb in labels]

        x = self.normalizer.fit_transform(x)
        self.model.fit(x, y)

    def inference(self, docs: List[Tuple[str, str]]) -> List[dict]:
        x = [self._transform(doc) for doc in docs]
        x = self.normalizer.transform(x)
        y_preds = self.model.predict_proba(x)

        possible_labels = self.map_labels.keys()
        return [dict(zip(possible_labels, y_pred)) for y_pred in y_preds]

    def predict(self, docs: List[Tuple[str, str]]) -> List[str]:
        inverse_mapper = {v: k for k, v in self.map_labels.items()}
        x = [self._transform(doc) for doc in docs]
        x = self.normalizer.transform(x)
        y_preds = self.model.predict(x)
        return [inverse_mapper[y] for y in y_preds]
