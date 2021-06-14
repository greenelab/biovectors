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

# # Subsample Document Experiment

# This notebook is designed to follow up on the decrease in token overlap between abstracts across the years. The hypothesis is that the small fraction of tokens shared across the years is due to the small number of available documents. To test theory documents posted in the years that have a high count are subsampled to be equal to the number of documents in the early most year (2000). After subsampling, the tokens are compared to see if the fraction of tokens shared changes.

# +
# %load_ext autoreload
# %autoreload 2

from collections import defaultdict, Counter
import csv
from datetime import datetime
import itertools
from pathlib import Path
import pickle

import lxml.etree as ET
import numpy as np
import pandas as pd
import plotnine as p9
import tqdm

from biovectors_modules.word2vec_run_helper import (
    PubMedSentencesIterator,
    PubtatorTarIterator,
    chunks,
)
# -

# # Load the Data

pickle_file = "output/unique_tokens_by_year_replace.pkl"
tokens_by_year = pickle.load(open(pickle_file, "rb"))

# # Compare Number of Tokens

# ## Equal Number of Tokens for each YEar

# +
data_rows = []
reversed_tokens = list(sorted(tokens_by_year.keys()))[::-1]
all_tokens = set(tokens_by_year[2020].keys())

for query_year in reversed_tokens[2:22]:
    np.random.seed(100)
    avail_tokens = list(tokens_by_year[query_year].keys())
    if len(avail_tokens) > len(all_tokens):
        avail_tokens = set(
            np.random.choice(avail_tokens, len(all_tokens), replace=False)
        )
        total_tokens = all_tokens
    else:
        total_tokens = set(
            np.random.choice(list(all_tokens), len(avail_tokens), replace=False)
        )

    query_year_vocab_set = set(avail_tokens)
    tokens_matched = total_tokens & query_year_vocab_set

    data_rows.append(
        {
            "years": str(query_year) if query_year != 2020 else "2020",
            "percentage_tokens_mapped": len(tokens_matched) / len(total_tokens),
            "num_tokens_matched": len(tokens_matched),
            "num_tokens_total": len(total_tokens),
        }
    )
# -

token_overlap_df = pd.DataFrame.from_dict(data_rows)
token_overlap_df

g = (
    p9.ggplot(
        token_overlap_df,
        p9.aes(x="years", y="percentage_tokens_mapped"),
    )
    + p9.geom_col(fill="#1f78b4")
    + p9.coord_flip()
    + p9.labs(
        title="Token Overlap across the Years (Subsampled)",
        x="Year",
        y="Fraction of Tokens Overlapped",
    )
)
g.save("output/figures/subsampled_tokens_overlap.png", dpi=500)
print(g)

# ## Equal number of tokens across each Year

# +
data_rows = []
reversed_tokens = list(sorted(tokens_by_year.keys()))[::-1]
np.random.seed(100)
all_tokens = np.random.choice(
    sorted(list(set(tokens_by_year[2020].keys()))), len(set(tokens_by_year[2000]))
)
all_tokens = set(all_tokens)

for query_year in reversed_tokens[2:22]:
    avail_tokens = list(tokens_by_year[query_year].keys())
    query_year_vocab_set = set(avail_tokens)
    tokens_matched = all_tokens & query_year_vocab_set

    data_rows.append(
        {
            "years": str(query_year) if query_year != 2020 else "2020",
            "percentage_tokens_mapped": len(tokens_matched) / len(total_tokens),
            "num_tokens_matched": len(tokens_matched),
            "num_tokens_total": len(total_tokens),
        }
    )
# -

equal_token_overlap_df = pd.DataFrame.from_dict(data_rows)
equal_token_overlap_df

g = (
    p9.ggplot(
        equal_token_overlap_df,
        p9.aes(x="years", y="percentage_tokens_mapped"),
    )
    + p9.geom_col(fill="#1f78b4")
    + p9.coord_flip()
    + p9.labs(
        title="Token Overlap across the Years (Subsampled)",
        x="Year",
        y="Fraction of Tokens Overlapped",
    )
)
g.save("output/figures/subsampled_tokens_overlap.png", dpi=500)
print(g)

# # Conclusions

# 1. The small number of mismatches appear to be the result of not using a lemmatizer along with general changes in scientific publications overtime.
# 2. Numbers could improve if I were to correct for the issues mentioned above; however, the overall message here is that earlier years have a bias that I nee
