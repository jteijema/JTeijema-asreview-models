import os
from urllib.request import urlretrieve
from pathlib import Path
import numpy as np
from gensim.models import KeyedVectors
from asreview.models.feature_extraction.base import BaseFeatureExtraction
from asreview.models.classifiers.base import BaseTrainClassifier
from asreview.utils import get_data_home
import fasttext
from scipy.sparse import csr_matrix

class FastTextFeatureExtractor(BaseFeatureExtraction):
    name = "fasttext"
    label = "FastText (crawl-300d-2M.vec)"

    EMBEDDING_EN = {
        "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip",
        "name": "crawl-300d-2M.vec",
    }

    @property
    def model(self):
        if not hasattr(self, "_model"):
            self._model = self.get_embedding_matrix()
            print("Embedding File loaded.")
        return self._model

    def get_embedding_matrix(self):
        data_home = Path(get_data_home())
        embedding_fp = data_home / self.EMBEDDING_EN["name"]
        print("\nLooking for embedding file in: ", embedding_fp)
        if not embedding_fp.exists():
            print("Embedding not found: Starting the download of the FastText embedding file.")
            self.download_embedding(data_home)
        else:
            print("Embedding file found.")
        print("Loading 4GB embedding file to memory.")
        return KeyedVectors.load_word2vec_format(embedding_fp, binary=False)

    def download_embedding(self, data_home):
        url = FastTextFeatureExtractor.EMBEDDING_EN["url"]
        file_name = FastTextFeatureExtractor.EMBEDDING_EN["name"]
        zip_path = data_home / (file_name + '.zip')
        print("Downloaded embedding file to: ", zip_path)
        urlretrieve(url, zip_path)
        print("Download complete.")

        print("Unzipping embedding file...")
        # Unzipping the file
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_home)
        print("Unzipping complete.")

        print("Removing zip file...")
        # Remove the zip file to save space
        os.remove(zip_path)
        print("Zip file removed.")

    def transform(self, texts):
        print("Encoding texts using FastText model...")
        transformed_texts = []
        for text in texts:
            transformed_texts.append(self.text_to_vector(text))
        print("Encoding complete.")
        self.clear_model()
        print("Unloading model.")
        return np.array(transformed_texts)

    def text_to_vector(self, text):
        words = text.split()
        word_vectors = [self.model[word] for word in words if word in self.model]
        if not word_vectors:
            return np.zeros(self.model.vector_size)
        return np.mean(word_vectors, axis=0)

    def clear_model(self):
        if hasattr(self, "_model"):
            del self._model


class FastTextClassifier(BaseTrainClassifier):
    """
    FastText Classifier extending the BaseTrainClassifier.
    """

    name = "fasttext"
    label = "FastText"

    def __init__(self):
        super().__init__()
        self._model = None

    def fit(self, X, y):
        """
        Fit the FastText model to the data.

        X: List[str]
            List of text samples.
        y: List[str]
            List of labels corresponding to each text sample.

        fit_kwargs: Additional keyword arguments for FastText training.
        """
        # Prepare training data in FastText format
        training_data_path = str(Path(get_data_home(), "training_data.txt"))
        with open(training_data_path, "w") as file:
            for text, label in zip(X, y):
                file.write(f"__label__{label} {text}\n")

        # Train the FastText model
        self._model = fasttext.train_supervised(input=training_data_path, verbose=0)

        # remove training_data
        os.remove(training_data_path)

    def predict_proba(self, X):
        """
        Get the inclusion probability for each sample.

        X: List[str] or similar iterable of strings
            List of text samples to predict.

        Returns
        -------
        numpy.ndarray
            Array with the probabilities for each class.
        """
        if self._model is None:
            raise ValueError("Model has not been fitted yet")

        probabilities = []

        # Ensure input is in the correct format
        if isinstance(X, csr_matrix):
            # Convert csr_matrix to a list of strings
            X = [" ".join(map(str, row)) for row in X.toarray()]

        for text in X:
            labels, probs = self._model.predict(text, k=2)  # k=2 for binary classification
            # Ensure correct order of probabilities
            if labels[0].endswith('1'):
                probabilities.append([probs[1], probs[0]])
            else:
                probabilities.append([probs[0], probs[1]])

        return np.array(probabilities)