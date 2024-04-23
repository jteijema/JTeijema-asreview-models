from asreview.models.feature_extraction.base import BaseFeatureExtraction
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings

class MXBAI(BaseFeatureExtraction):
    name = "mxbai"
    label = "mixedbread ai feature extraction"

    def __init__(self, model_name="mixedbread-ai/mxbai-embed-large-v1", quantize=False, precision="ubinary"):
        super().__init__()
        self.model_name = model_name
        self.quantize = quantize
        self.precision = precision

    @property
    def _model(self):
        if not hasattr(self, "model"):
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def transform(self, texts):
        embeddings = self._model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        if self.quantize:
            embeddings = quantize_embeddings(embeddings, precision=self.precision)
        return embeddings.numpy()
