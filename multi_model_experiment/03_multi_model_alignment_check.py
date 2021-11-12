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

# # Calculate Statistics for Multiple Models per Year

# Word2Vec is an unstable model as they arbitrarily generate word vectors [10.1007/s00799-019-00271-6](http://doi.org/10.1007/s00799-019-00271-6).
# This means word vectors from two different word2vec models cannot be compared directly.
# A way to fix this issue is to perform a task called alignment, which involves aligning multiple word2vec models onto the same vector space.
# The approach to align word vector models is called orthogonal procrustes, which is a linear algebra technique that aligns two vector spaces while preserving the orthogonal property of vectors.
# This notebook designed to confirm that the alignment step works for these models.

# +
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

from gensim.models import Word2Vec, KeyedVectors
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import plotnine as p9
from plydata import define, select, group_by, do, ungroup, query, arrange
import tqdm
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP

from biovectors_modules.word2vec_analysis_helper import align_word2vec_models
# -

# # Load Word2Vec model

word_models = list(Path("output/models").rglob("*model"))
word_models = sorted(word_models, key=lambda x: x.stem)
word_model_filter = list(filter(lambda x: "2001" in x.stem, word_models))

words_to_visualize = []
most_frequent_tokens = []

# +
for model_file in tqdm.tqdm(word_model_filter):
    word_model = Word2Vec.load(str(model_file))
    most_frequent_tokens.append(
        {word: word_model.wv[word] for word in word_model.wv.index_to_key[:100]}
    )
    words_to_visualize.append(np.array(word_model.wv.vectors))

words_to_visualize = np.vstack(words_to_visualize)
# -

# # Train the UMAP model on Unaligned Vectors

file_name = "output/umap_models/2001_no_alignment"
tokens = list(most_frequent_tokens[0].keys())
words_subset = np.vstack(
    list(map(lambda x: np.vstack(list(x.values())), most_frequent_tokens))
)

if not Path(file_name).exists():
    model = ParametricUMAP(
        verbose=False,
        metric="cosine",
        random_state=100,
        low_memory=True,
        n_neighbors=25,
        min_dist=0.0,
    )
    model.fit(words_to_visualize)
    embedding = model.transform(words_subset)
    model.save(file_name)
else:
    model = load_ParametricUMAP(file_name)
    embedding = model.transform(words_subset)

mapped_df = (
    pd.DataFrame(embedding, columns=["umap1", "umap2"])
    >> define(token=tokens * 10)
    >> define(
        model_iter=sorted(list(range(10)) * len(tokens))
    )  # copy each model iter for each token num
)
mapped_df.model_iter = pd.Categorical(mapped_df.model_iter)
mapped_df.head()

g = (
    p9.ggplot(mapped_df, p9.aes(x="umap1", y="umap2", fill="model_iter"))
    + p9.geom_point()
    + p9.labs(title="100 Most Frequently Occuring Tokens in 2001")
)
print(g)

# # Train UMAP Model on Aligned Vectors

base_model = Word2Vec.load(str(word_models[-1]))
words_to_visualize = []
most_frequent_tokens = []

# +
for model_file in tqdm.tqdm(word_model_filter):
    word_model = Word2Vec.load(str(model_file))
    aligned_model = align_word2vec_models(base_model, word_model)
    most_frequent_tokens.append(
        {word: aligned_model.wv[word] for word in aligned_model.wv.index_to_key[:100]}
    )
    words_to_visualize.append(np.array(aligned_model.wv.vectors))

words_to_visualize = np.vstack(words_to_visualize)
# -

file_name = "output/umap_models/2001_with_alignment"
tokens = list(most_frequent_tokens[0].keys())
words_subset = np.vstack(
    list(map(lambda x: np.vstack(list(x.values())), most_frequent_tokens))
)

if not Path(file_name).exists():
    model = ParametricUMAP(
        verbose=False,
        metric="cosine",
        random_state=100,
        low_memory=True,
        n_neighbors=25,
        min_dist=0.0,
    )
    model.fit(words_to_visualize)
    embedding = model.transform(words_subset)
    model.save(file_name)
else:
    model = load_ParametricUMAP(file_name)
    embedding = model.transform(words_subset)

mapped_df = (
    pd.DataFrame(embedding, columns=["umap1", "umap2"])
    >> define(token=tokens * 10)
    >> define(
        model_iter=sorted(list(range(10)) * len(tokens))
    )  # copy each model iter for each token num
)
mapped_df.model_iter = pd.Categorical(mapped_df.model_iter)
mapped_df.head()

g = (
    p9.ggplot(mapped_df, p9.aes(x="umap1", y="umap2", fill="model_iter"))
    + p9.geom_point()
    + p9.labs(title="100 Most Frequently Occuring Tokens in 2001")
)
print(g)

g = (
    p9.ggplot(
        mapped_df >> define(opacity=lambda x: x.token == "the"),
        p9.aes(x="umap1", y="umap2", fill="model_iter", alpha="opacity"),
    )
    + p9.geom_point()
    + p9.labs(title="the in 2001")
    + p9.guides(alpha=False)
)
print(g)

g = (
    p9.ggplot(
        mapped_df >> define(opacity=lambda x: x.token == "of"),
        p9.aes(x="umap1", y="umap2", fill="model_iter", alpha="opacity"),
    )
    + p9.geom_point()
    + p9.labs(title="'of' in 2001")
    + p9.guides(alpha=False)
)
print(g)
