# README for ASReview-experimental-models: Comprehensive Model Suite
## Overview

This plugin for ASReview provides a diverse suite of models, expanding the
capabilities of the ASReview software for automated systematic reviews. The
plugin integrates several advanced classifiers and feature extraction
techniques, enabling users to leverage a broad range of algorithms for their
systematic review processes.

> ! HIGHLY EXPERIMENTAL !

### Included Models

XGBoost: A highly efficient and scalable implementation of gradient boosting.

ALL-MPNet-Base-v2 (Hier. Mean): An advanced transformer-based model optimized for
semantic understanding with hierarchical mean pooling. Note, max sequence length
is 384.

DistilUSE-Base-Multilingual-Cased-v2: A multilingual transformer-based model,
offering robust performance across various languages. Note, max sequence length
is 128.

LaBSE: Language-agnostic BERT Sentence Embedding model, excellent for semantic
similarity and retrieval tasks in multiple languages. Note, max sequence length
is 256.

OneHot: A simple baseline model that uses one-hot encoding for feature
extraction.

FastText: A powerful text representation and classification model, particularly
effective for tasks involving large vocabularies and rich text data. Uses
`wiki-news-300d-1M-subword.vec`.

Convolutional Neural Network: A deep learning model that uses convolutional and
pooling layers to extract features from text data.

Doc2Vec (120 vector version): A powerful text representation and classification
model, particularly effective for tasks involving large vocabularies and rich
text data.

Spacy: Uses `en_core_web_md` model from Spacy to extract features from text data.

Word2Vec: A powerful text representation and classification model, based on
`word2vec-google-news-300`.

Dynamic_nn: A neural network model that uses a dynamic architecture. Every
factor 10 increase in the number of samples, the number of layers is increased.

KNN: A simple baseline model that uses K-Nearest Neighbors for classification.

AdaBoost: A simple baseline model that uses AdaBoost for classification.

Random: A simple baseline model that uses random classification.

### Keywords
Classifiers Available:

    xgboost
    power_cnn
    dynamic_nn
    adaboost
    knn
    random

Feature Extractors Available:

    ft-sbert
    wide_doc2vec
    multilingual
    labse
    fasttext
    onehot
    spacy
    word2vec
    scaled-doc2vec
    softmax-doc2vec
    relu-doc2vec

## Installation

To install this plugin, use one of the following method:
```bash
pip install git+https://github.com/jteijema/ASReview-experimental-models.git
```

## Usage

Once installed, the models from this plugin can be used in ASReview simulations.
For example, to use the XGBoost model, run:

```bash
asreview simulate example_data_file.csv -m xgboost
```

Replace xgboost with the appropriate model identifier to use other models.

## Compatibility

This plugin is compatible with the latest version of ASReview. Ensure that your
ASReview installation is up-to-date to avoid compatibility issues.

## License

This ASReview plugin is released under the MIT License. See the LICENSE file for
more details.
