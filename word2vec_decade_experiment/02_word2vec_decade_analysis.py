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

# # Analyze Word2Vec by Decades Run

# +
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import pickle
import itertools

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import orthogonal_procrustes
import tqdm
import plotnine as p9

from biovectors_modules.word2vec_analysis_helper import (
    get_global_distance,
    get_local_distance,
    window,
)
# -

# ## Load Models and Parse Performance

# Align word2vec Models since cutoff year
year_cutoff = 2000
aligned_model_file_path = "output/aligned_word_vectors.pkl"
token_occurence_file = "output/earliest_token_occurence.tsv"
year_distance_folder = "year_distances"

# Skip 2021 as that model is too small to analyze
# Try again December 2021
word_models = filter(
    lambda x: int(x.stem.split("_")[1]) >= year_cutoff
    and int(x.stem.split("_")[1]) != 2021,
    list(Path("output/models").rglob("*model")),
)
word_models = sorted(word_models, key=lambda x: int(x.stem.split("_")[1]), reverse=True)
print(word_models)

if not Path(token_occurence_file).exists():
    word_model_dict = dict()
    earliest_token_occurence = dict()
    for model in reversed(word_models):
        year = int(model.stem.split("_")[1])
        model = Word2Vec.load(str(model))
        for token in model.wv.vocab.keys():
            if token not in earliest_token_occurence:
                earliest_token_occurence[token] = f"{year}"
            else:
                earliest_token_occurence[token] += f"|{year}"
        (
            pd.DataFrame(
                list(earliest_token_occurence.items()),
                columns=["token", "year_occured"],
            ).to_csv(token_occurence_file, sep="\t", index=False)
        )

if not Path(aligned_model_file_path).exists():
    word_model_dict = dict()
    shared_tokens = set()
    for model in word_models:
        year = int(model.stem.split("_")[1])
        word_model_dict[year] = Word2Vec.load(str(model))
        if len(shared_tokens) == 0:
            shared_tokens = set(word_model_dict[year].wv.vocab.keys())
        else:
            shared_tokens &= set(word_model_dict[year].wv.vocab.keys())

    shared_tokens = sorted(list(shared_tokens))

# ## Calculate Global and Local Distances Between Time Periods

# ### Align Models via Orthogonal Procrustes

if not Path(aligned_model_file_path).exists():
    years_analyzed = sorted(list(word_model_dict.keys()), reverse=True)
    latest_year = years_analyzed[0]
    aligned_models = {}

    # Years must be in sorted descending order
    for year_pair in window(years_analyzed):

        year_pair_label = "_".join(map(str, year_pair[::-1]))

        if year_pair[0] == latest_year:
            aligned_models[str(year_pair[0])] = word_model_dict[year_pair[0]].wv[
                shared_tokens
            ]

        else:

            aligned_models[str(year_pair[0])] = word_model_dict[year_pair[0]].wv[
                shared_tokens
            ]

            translation_matrix, scale = orthogonal_procrustes(
                word_model_dict[year_pair[0]].wv[shared_tokens],
                word_model_dict[latest_year].wv[shared_tokens],
            )

        translation_matrix, scale = orthogonal_procrustes(
            word_model_dict[year_pair[1]].wv[shared_tokens],
            word_model_dict[latest_year].wv[shared_tokens],
        )

        aligned_models[str(year_pair[1])] = (
            word_model_dict[year_pair[1]].wv[shared_tokens] @ translation_matrix
        )

    aligned_models["shared_tokens"] = shared_tokens

if not Path(aligned_model_file_path).exists():
    pickle.dump(aligned_models, open(aligned_model_file_path, "wb"))

# ### Calculate the Global and Local Distances between Words

aligned_models = pickle.load(open(aligned_model_file_path, "rb"))
years_analyzed = sorted(list(aligned_models.keys()), reverse=True)[1:]
n_neighbors = 25

shared_tokens = sorted(aligned_models["shared_tokens"])
for key in tqdm.tqdm(window(years_analyzed)):

    global_distance = get_global_distance(
        aligned_models[key[0]], aligned_models[key[1]], aligned_models["shared_tokens"]
    )

    local_distance = get_local_distance(
        aligned_models[key[0]],
        aligned_models[key[1]],
        aligned_models["shared_tokens"],
        neighbors=n_neighbors,
    )

    label = "_".join(reversed(key))

    (
        global_distance.merge(local_distance)
        .assign(shift=lambda x: x.global_dist.values - x.local_dist.values)
        .to_csv(
            f"output/{year_distance_folder}/{label}_dist.tsv", sep="\t", index=False
        )
    )
