# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # Look into changepoints detected by CUSUM

# +
import itertools
from pathlib import Path

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import plydata as ply
import tqdm

from detecta import detect_cusum

from biovectors_modules.plot_helper import deidentify_concepts, generate_neighbor_table
# -

output_file_folder = Path("output/figure_data_and_figures/neighbor_tables")

word2vec_model_list = sorted(
    list(Path("output/models").rglob("*0_fulltext.model")),
    key=lambda x: int(x.stem.split("_")[0]),
)
print(word2vec_model_list[0:2])

concepts_df = pd.read_csv(
    "../concept_mapper_experiment/output/all_concept_ids.tsv.xz", sep="\t"
)
lower_case_concept_id = list(map(lambda x: x.lower(), concepts_df.concept_id.tolist()))
concept_mapper = dict(zip(lower_case_concept_id, concepts_df.concept.tolist()))
concepts_df >> ply.slice_rows(10)

changepoints_df = pd.read_csv("output/pubtator_updated_changepoints.tsv", sep="\t")
changepoints_df

query = "reviewer(s"
neighbor_thru_time_df = generate_neighbor_table(
    word2vec_model_list, query, changepoints_df, concept_mapper, n_neighbors=10
)
neighbor_thru_time_df.T

query = "medrxiv"
neighbor_thru_time_df = generate_neighbor_table(
    word2vec_model_list,
    query,
    changepoints_df,
    concept_mapper,
    n_neighbors=10,
    save_file=True,
)
neighbor_thru_time_df.T

query = "pandemic"
neighbor_thru_time_df = generate_neighbor_table(
    word2vec_model_list,
    query,
    changepoints_df,
    concept_mapper,
    n_neighbors=10,
    save_file=True,
)
neighbor_thru_time_df.T

query = "cas9"
neighbor_thru_time_df = generate_neighbor_table(
    word2vec_model_list,
    query,
    changepoints_df,
    concept_mapper,
    n_neighbors=10,
    save_file=True,
)
neighbor_thru_time_df.T

query = "cellline_cvcl_1698"
neighbor_thru_time_df = generate_neighbor_table(
    word2vec_model_list, query, changepoints_df, concept_mapper, n_neighbors=10
)
neighbor_thru_time_df.T
