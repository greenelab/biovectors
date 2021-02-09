# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1+dev
#   kernelspec:
#     display_name: Python [conda env:biovectors]
#     language: python
#     name: conda-env-biovectors-py
# ---

# # Run Word2Vec to predict Disease Gene Paris

# +
from pathlib import Path

from gensim.models import Word2Vec, KeyedVectors
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import os

from biovectors_modules.word2vec_run_helper import (
    get_gene_disease_pairs,
    Sentences,
    similarity_score,
)
# -

# ## Set up the Data

pairs = get_gene_disease_pairs(
    Path("output/hetnet_gene_disease_pairs.tsv"), Path("output/DO-slim-to-mesh.tsv")
)
pairs[0:10]

# ## Fire up word2vec

# Set up the path to abstract sentences
sentences = Sentences(Path("inputs/bioconcepts2pubtatorcentral.gz"))

model = Word2Vec(sentences, size=500, window=5, min_count=1, workers=4)

model.save(Path("output/word2vec.model"))

# ## Get Similarity Scores

similarity_scores_df = similarity_score(model, pairs)
similarity_scores_df.head()

similarity_scores_df.to_csv("output/similarity_scores.tsv", sep="\t", index=False)
