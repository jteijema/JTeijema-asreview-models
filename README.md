# README for ASReview Extension Plugin: Comprehensive Model Suite
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
offering robust performance across various languages.

LaBSE: Language-agnostic BERT Sentence Embedding model, excellent for semantic
similarity and retrieval tasks in multiple languages. Note, max sequence length
is 256.

FastText: A powerful text representation and classification model, particularly
effective for tasks involving large vocabularies and rich text data.

Word2Vec + DAN (Deep Averaging Network): Combines the Word2Vec embeddings with a
Deep Averaging Network for effective text classification.

## Installation

To install this plugin, use one of the following method:
```bash
pip install git+https://github.com/jteijema/JTeijema-asreview-models.git
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

This ASReview plugin is released under the MIT License. See the LICENSE file for more details.