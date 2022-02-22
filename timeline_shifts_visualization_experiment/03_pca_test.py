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

# # Test if PCA can separate the unique word2vec models

# This notebook is designed to see if PCA can test apart unaligned word2vec models.

# +
import itertools
from pathlib import Path
import re

from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pandas as pd
import plotnine as p9
import plydata as ply
import plydata.tidy as ply_tdy
from sklearn.decomposition import PCA
import tqdm
import umap
# -

# # Unaligned Models

# ## Load the paths for each year

# The goal here is to grab the odd years unaligned and see if umap can tease apart odd years.

unaligned_word_models = list(
    Path("../multi_model_experiment/output/models").rglob("*model")
)
unaligned_word_models = sorted(unaligned_word_models, key=lambda x: x.stem)
unaligned_word_model_filter = list(
    filter(lambda x: "2001" in x.stem, unaligned_word_models)
)

word_freq_count_cutoff = 5
odd_year_subset = unaligned_word_model_filter

training_unaligned_word_model_map = dict()
for word_file in tqdm.tqdm(odd_year_subset):
    model = Word2Vec.load(str(word_file)).wv
    training_unaligned_word_model_map[word_file.stem] = dict(
        model=model,
        cutoff_index=min(
            map(
                lambda x: 999999
                if model.get_vecattr(x[1], "count") > word_freq_count_cutoff
                else x[0],
                enumerate(model.index_to_key),
            )
        ),
    )

words_to_visualize = []
token_character_list = []
year_labels_list = []

for year in tqdm.tqdm(training_unaligned_word_model_map):
    model = training_unaligned_word_model_map[year]["model"]
    word_subset_matrix = model[
        model.index_to_key[: training_unaligned_word_model_map[year]["cutoff_index"]]
    ]
    print((year, word_subset_matrix.shape))
    words_to_visualize.append(word_subset_matrix)
    token_character_list += list(
        map(
            lambda x: re.escape(x),
            model.index_to_key[
                : training_unaligned_word_model_map[year]["cutoff_index"]
            ],
        )
    )
    year_labels_list += [year] * len(
        model.index_to_key[: training_unaligned_word_model_map[year]["cutoff_index"]]
    )

training_unaligned_words = np.vstack(words_to_visualize)
training_unaligned_words

# ## Run PCA

model = PCA(n_components=2, random_state=100)
embeddings = model.fit_transform(training_unaligned_words)
transformed_df = pd.DataFrame(embeddings, columns=["pca1", "pca2"]) >> ply.define(
    year=year_labels_list, tok=token_character_list
)
transformed_df >> ply.slice_rows(10)

g = (
    p9.ggplot(
        transformed_df >> ply.define(year="pd.Categorical(year)"),
        p9.aes(x="pca1", y="pca2", fill="year"),
    )
    + p9.geom_point(alpha=0.2)
    + p9.labs(title="PCA of Unaligned 2001")
    + p9.theme(figure_size=(10, 8))
)
print(g)

model = umap.parametric_umap.ParametricUMAP(
    verbose=True,
    metric="cosine",
    random_state=100,
    low_memory=True,
    n_neighbors=25,
    min_dist=0.0,
)
embedding = model.fit_transform(training_unaligned_words)

transformed_umap_df = pd.DataFrame(embedding, columns=["umap1", "umap2"]) >> ply.define(
    year=year_labels_list, tok=token_character_list
)
transformed_umap_df >> ply.slice_rows(10)

g = (
    p9.ggplot(
        transformed_umap_df >> ply.define(year="pd.Categorical(year)"),
        p9.aes(x="umap1", y="umap2", fill="year"),
    )
    + p9.geom_point(alpha=0.2)
    + p9.labs(title="UMAP of Unaligned 2001")
    + p9.theme(figure_size=(10, 8))
)
print(g)
