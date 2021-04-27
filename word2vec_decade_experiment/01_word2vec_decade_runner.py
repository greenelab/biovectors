#!/usr/bin/env python
# coding: utf-8
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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

# # Run Word2vec on abstracts for each Decade

# +
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

from gensim.models import Word2Vec, KeyedVectors
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import re
import tqdm

from biovectors_modules.word2vec_run_helper import PubMedSentencesIterator, chunks
# -

# # Set up the Data

pubtator_central_metadata_df = pd.read_csv(
    "../exploratory_data_analysis/output/pmc_metadata.tsv.xz", sep="\t"
)
print(pubtator_central_metadata_df.shape)
pubtator_central_metadata_df.head()


pubtator_abstract_batch = list(Path("../pubtator_abstracts").rglob("*.gz"))
print(len(pubtator_abstract_batch))

year_mapper_df = (
    pubtator_central_metadata_df.query("published_year != 0")
    .groupby(["published_year", "batch_folder"])
    .agg({"batch_file": "unique"})
    .reset_index()
)
year_mapper_df.head()

batch_file_year_dict = {}
for idx, row in tqdm.tqdm(year_mapper_df.iterrows()):
    if row["published_year"] not in batch_file_year_dict:
        batch_file_year_dict[row["published_year"]] = {}

    batch_file_year_dict[row["published_year"]][row["batch_folder"]] = list(
        row["batch_file"]
    )

# # Run the Models

# Generate path to save word2vec models
Path("output/models").mkdir(exist_ok=True)
years = (
    pubtator_central_metadata_df.query("published_year>=2000")
    .published_year.unique()
    .tolist()
)


# iterate through all abstracts through all years
random.seed(100)
for idx, year in enumerate(chunks(sorted(years), 1)):

    if Path(f"output/models/word2vec_{str(year)}.model").exists():
        continue

    random.shuffle(pubtator_abstract_batch)

    doc_iterator = PubMedSentencesIterator(
        pubtator_abstract_batch,
        batch_mapper=batch_file_year_dict[year[0]],
        year_filter=year,
        jobs=40,
    )

    model = Word2Vec(size=300, window=5, min_count=1, workers=8, seed=100)
    model.build_vocab(doc_iterator)

    model.train(doc_iterator, epochs=5, total_examples=model.corpus_count)

    model.save(str(Path(f"output/models/word2vec_{str(year[0])}.model")))

    print(f"Saved {str(year[0])} word2vec model")

