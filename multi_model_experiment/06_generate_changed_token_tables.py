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

# # Generate Tables of the Most Changed Tokens

# +
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotnine as p9
import plydata as ply
import plydata.tidy as ply_tdy
import tqdm
# -

subsetted_tokens = pd.read_csv("output/subsetted_tokens.tsv", sep="\t")
token_filter_list = subsetted_tokens.tok.tolist()
subsetted_tokens.head()

distance_files = list(
    Path("output/combined_inter_intra_distances").rglob("saved_*_distance.tsv")
)
print(len(distance_files))

year_distance_map = {
    re.search(r"\d+", str(year_file)).group(0): (pd.read_csv(str(year_file), sep="\t"))
    for year_file in tqdm.tqdm(distance_files)
}

full_token_set_df = pd.concat(
    [
        year_distance_map[year] >> ply.query(f"tok in {token_filter_list}")
        # >>ply.query("year_2-year_1 == 1")
        >> ply.query("year_1 == 2000")
        for year in tqdm.tqdm(year_distance_map)
    ]
)
print(full_token_set_df.shape)
full_token_set_df.head()

# # Top Ten Words that has the Greatest Rate of Change in X time

# ## 20 Years

(
    full_token_set_df
    >> ply.query("year_2==2020")
    >> ply.arrange("-global_distance_qst")
    >> ply.slice_rows(10)
)

(
    full_token_set_df
    >> ply.query("year_2==2020")
    >> ply.arrange("-original_global_distance")
    >> ply.slice_rows(10)
)

(
    full_token_set_df
    >> ply.query("year_2==2020")
    >> ply.arrange("-global_times_distance_qst")
    >> ply.slice_rows(10)
)

# ## 10 Years

(
    full_token_set_df
    >> ply.query("year_2==2010")
    >> ply.arrange("-global_distance_qst")
    >> ply.slice_rows(10)
)

(
    full_token_set_df
    >> ply.query("year_2==2010")
    >> ply.arrange("-original_global_distance")
    >> ply.slice_rows(10)
)

(
    full_token_set_df
    >> ply.query("year_2==2010")
    >> ply.arrange("-global_times_distance_qst")
    >> ply.slice_rows(10)
)

# ## 5 Years

(
    full_token_set_df
    >> ply.query("year_2==2005")
    >> ply.arrange("-global_distance_qst")
    >> ply.slice_rows(10)
)

(
    full_token_set_df
    >> ply.query("year_2==2005")
    >> ply.arrange("-original_global_distance")
    >> ply.slice_rows(10)
)

(
    full_token_set_df
    >> ply.query("year_2==2005")
    >> ply.arrange("-global_times_distance_qst")
    >> ply.slice_rows(10)
)
