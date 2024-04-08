import numpy as np

from gensim.models.doc2vec import Doc2Vec as GenSimDoc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess

from asreview.models.feature_extraction.base import BaseFeatureExtraction


def _train_model(corpus, *args, **kwargs):
    model = GenSimDoc2Vec(*args, **kwargs)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def _transform_text(model, corpus):
    X = []
    for doc_id in range(len(corpus)):
        doc_vec = model.infer_vector(corpus[doc_id].words)
        X.append(doc_vec)
    return np.array(X)


class ReLUDoc2Vec(BaseFeatureExtraction):
    name = "relu-doc2vec"
    label = "ReLUDoc2Vec"

    def __init__(
        self,
        *args,
        vector_size=40,
        epochs=33,
        min_count=1,
        n_jobs=1,
        window=7,
        dm_concat=0,
        dm=2,
        dbow_words=0,
        **kwargs
    ):
        """Initialize the doc2vec model."""
        super().__init__(*args, **kwargs)
        self.vector_size = int(vector_size)
        self.epochs = int(epochs)
        self.min_count = int(min_count)
        self.n_jobs = int(n_jobs)
        self.window = int(window)
        self.dm_concat = int(dm_concat)
        self.dm = int(dm)
        self.dbow_words = int(dbow_words)
        self._model = None
        self._model_dm = None
        self._model_dbow = None

    def fit(self, texts):
        model_param = {
            "vector_size": self.vector_size,
            "epochs": self.epochs,
            "min_count": self.min_count,
            "workers": self.n_jobs,
            "window": self.window,
            "dm_concat": self.dm_concat,
            "dbow_words": self.dbow_words,
        }

        corpus = [
            TaggedDocument(simple_preprocess(text), [i]) for i, text in enumerate(texts)
        ]

        if self.dm == 2:
            model_param["vector_size"] = int(model_param["vector_size"] / 2)
            self.model_dm = _train_model(corpus, **model_param, dm=1)
            self.model_dbow = _train_model(corpus, **model_param, dm=0)
        else:
            self.model = _train_model(corpus, **model_param, dm=self.dm)

    def transform(self, texts):
        corpus = [
            TaggedDocument(simple_preprocess(text), [i]) for i, text in enumerate(texts)
        ]

        if self.dm == 2:
            X_dm = _transform_text(self.model_dm, corpus)
            X_dbow = _transform_text(self.model_dbow, corpus)
            X = np.concatenate((X_dm, X_dbow), axis=1)
        else:
            X = _transform_text(self.model, corpus)

        X = np.maximum(0, X)
        return X
