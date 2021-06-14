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

# This notebook is designed to calculate statistics on fully trained word2vec models trained in [01_word2vec_decade_runner.ipynb](01_word2vec_decade_runner.ipynb). The statistics calculated are the cosine distance between tokens on a global level and a local level. Cosine distance is a helpful metric as it isn't affected by the magnitude of vectors.

# +
# %load_ext autoreload
# %autoreload 2

from collections import defaultdict
import csv
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

from biovectors_modules.word2vec_analysis_helper import get_global_local_distance
# -

# ## Load Models and Parse Performance

# Align word2vec Models since cutoff year
year_cutoff = 2000
latest_year = 2020
aligned_model_file_path = f"output/aligned_word_vectors_{year_cutoff}_{latest_year}.pkl"
aligned_models = dict()
token_occurence_file = "output/earliest_token_occurence.tsv"

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
    earliest_token_occurence = dict()
    for model in reversed(word_models):
        year = model.stem.split("_")[1]
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

# ## Calculate Global and Local Distances Between Time Periods

# ### Align Models via Orthogonal Procrustes

if not Path(aligned_model_file_path).exists():

    for model in tqdm.tqdm(word_models):
        year = model.stem.split("_")[1]
        word_model = Word2Vec.load(str(model))

        if year == str(latest_year):
            origin_tokens = sorted(list(set(word_model.wv.vocab.keys())))
            remaining_tokens = origin_tokens

        else:
            tokens = sorted(list(set(word_model.wv.vocab.keys())))
            remaining_tokens = set(origin_tokens) & set(tokens)

        data_records = []

        for tok in remaining_tokens:
            data_entry = dict(
                zip(
                    [f"feat_{idx}" for idx in range(len(word_model.wv[tok]))],
                    word_model.wv[tok],
                )
            )
            data_entry["token"] = tok
            data_records.append(data_entry)

        aligned_models[year] = pd.DataFrame.from_records(data_records)

if not Path(aligned_model_file_path).exists():
    years_analyzed = sorted(list(aligned_models.keys()), reverse=True)[1:]
    latest_year = str(latest_year)

    origin_df = aligned_models[latest_year].set_index("token")

    # Years must be in sorted descending order
    for year in tqdm.tqdm(years_analyzed):
        tokens = sorted(aligned_models[year].token.tolist())
        needs_aligned_df = aligned_models[year].set_index("token")

        # align A to B subject to transition matrix being
        # orthogonal to preserve the cosine similarities
        translation_matrix, scale = orthogonal_procrustes(
            needs_aligned_df.loc[tokens].values,
            origin_df.loc[tokens].values,
        )

        # Matrix Multiplication to project year onto 2020
        aligned_word_matrix = needs_aligned_df.loc[tokens].values @ translation_matrix

        corrected_df = pd.DataFrame(aligned_word_matrix)
        corrected_df.columns = needs_aligned_df.columns.tolist()
        aligned_models[year] = corrected_df.assign(token=tokens)

if not Path(aligned_model_file_path).exists():
    pickle.dump(aligned_models, open(aligned_model_file_path, "wb"))

# ### Calculate the Global and Local Distances between Words

aligned_models = pickle.load(open(aligned_model_file_path, "rb"))
n_neighbors = 25
year_distance_file = "output/all_year_distances.tsv"

earliest_dict = defaultdict(list)
start_stop_years = dict()
start = 0
word_vectors = []

for year in tqdm.tqdm(sorted(aligned_models.keys())):
    # Get token occurence
    for idx, entry in enumerate(
        aligned_models[year].sort_values("token").token.tolist()
    ):
        earliest_dict[entry].append((year, idx + start))

    # Get the year indicies
    start_stop_years[year] = (start, start + aligned_models[year].shape[0])
    start += aligned_models[year].shape[0]
    word_vectors.append(
        aligned_models[year].sort_values("token").set_index("token").values
    )
global_word_vectors = np.vstack(word_vectors)

get_global_local_distance(
    global_word_vectors,
    earliest_dict,
    start_stop_years,
    neighbors=25,
    n_jobs=3,
    output_file=year_distance_file,
)
