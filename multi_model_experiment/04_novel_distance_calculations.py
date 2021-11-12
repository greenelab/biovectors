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

# # Statistical Test for Multi-Model Variation

# +
# %load_ext autoreload
# %autoreload 2

from collections import Counter
import csv
import copy
import itertools
import math
from pathlib import Path
import random
import re

from gensim.models import Word2Vec, KeyedVectors
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import plotnine as p9
import plydata as ply
import plydata.tidy as ply_tdy
import scipy.stats as stats
import tqdm

from biovectors_modules.word2vec_analysis_helper import align_word2vec_models


# -

# Method only used for this notebook
def return_global_plot(year_model, tok="are", limits=(0, 1), inter_or_intra="intra"):
    g = (
        p9.ggplot(
            year_model >> ply.query(f"tok=='{tok}'"),
            p9.aes(x="year", y="global_distance"),
        )
        + p9.geom_boxplot()
        + p9.labs(
            title=f"{inter_or_intra.capitalize()} Year global Distance for Token: '{tok}'"
        )
        + p9.coord_flip()
        + p9.scale_y_continuous(limits=limits)
        + p9.theme_seaborn(style="white")
    )
    return g


# # Grab a listing of all word models

word_models = list(Path("output/models").rglob("*model"))
word_models = sorted(word_models, key=lambda x: x.stem)
word_model_filter = list(filter(lambda x: "2021" not in x.stem, word_models))

alignment_base_model = Word2Vec.load(str(word_model_filter[-1]))
temp_output_path = Path("output/aligned_vectors_tmp")

for model_file in tqdm.tqdm(word_model_filter):
    if not Path(f"{str(temp_output_path)}/{model_file.stem}.kv").exists():
        word_model = Word2Vec.load(str(model_file))
        aligned_model = align_word2vec_models(alignment_base_model.wv, word_model.wv)
        aligned_model.save(f"{str(temp_output_path)}/{model_file.stem}.kv")

# # Inter and Intra Variation calculation

# Refer to the following scripts in order to perform inter and intra word2vec calculations:
# 1. pmacs_cluster_running_inter_model_variation.py
# 2. pmacs_cluster_running_intra_model_variation.py

# # Are word2vec models unstable?

# Due to the nature of negative sampling word2vec models generate weights arbitrarily.
# This is undesired as a token in the year 2000 cannot be compared with a token in 2001.
# A solution is to use orthogonal procrustes to align word2vec models; however, variation could still remain in these word models.
# To measure this variation I trained 10 unique word2vec models on abstracts for each given year and then calculated global and local distances between every word2vec model pair (10 choose 2).
# From there I analyzed variation within each year (term intra-year variation).

# ## Intra Model Calculations

intra_year_models = []
for idx, file in enumerate(Path("output/intra_models").rglob("*.tsv.xz")):
    print(file)
    intra_year_model_df = pd.read_csv(str(file), sep="\t") >> ply_tdy.extract(
        "year_pair", into="year", regex=r"(\d+)_", convert=True
    )

    intra_year_models.append(intra_year_model_df)

    if Path(
        f"output/averaged_intra_models/average_{str(Path(file.stem).stem)}.tsv"
    ).exists():
        continue

    averaged_intra_year_models = dict()
    for idx, row in tqdm.tqdm(
        intra_year_model_df.iterrows(), desc=f"intra_df: {str(file)}"
    ):
        if (row["tok"], int(row["year"])) not in averaged_intra_year_models:
            averaged_intra_year_models[(row["tok"], int(row["year"]))] = dict(
                global_distance=[], local_distance=[]
            )

        averaged_intra_year_models[(row["tok"], int(row["year"]))][
            "global_distance"
        ].append(row["global_distance"])
        averaged_intra_year_models[(row["tok"], int(row["year"]))][
            "local_distance"
        ].append(row["local_distance"])

    with open(
        f"output/averaged_intra_models/average_{str(Path(file.stem).stem)}.tsv", "w"
    ) as outfile:
        fieldnames = [
            "average_global_distance",
            "average_local_distance",
            "var_global_distance",
            "var_local_distance",
            "tok",
            "year",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for tok, year in tqdm.tqdm(
            averaged_intra_year_models, desc=f"summary_intra_writer: {str(file.stem)}"
        ):
            writer.writerow(
                {
                    "average_global_distance": np.mean(
                        averaged_intra_year_models[(tok, year)]["global_distance"]
                    ),
                    "var_global_distance": np.var(
                        averaged_intra_year_models[(tok, year)]["global_distance"]
                    ),
                    "average_local_distance": np.mean(
                        averaged_intra_year_models[(tok, year)]["local_distance"]
                    ),
                    "var_local_distance": np.var(
                        averaged_intra_year_models[(tok, year)]["local_distance"]
                    ),
                    "tok": tok,
                    "year": year,
                }
            )

intra_year_models = pd.concat(intra_year_models)
intra_year_models.year = pd.Categorical(intra_year_models.year.tolist())
intra_year_models.head()

return_global_plot(intra_year_models, limits=(0, 0.1))

return_global_plot(intra_year_models, "privacy", limits=(0, 0.5))

return_global_plot(intra_year_models, "rna", limits=(0, 0.5))

# ## Inter Model Calculations

for idx, file in enumerate(Path("inter_models").rglob("*.tsv")):
    inter_year_model_df = pd.read_csv(str(file), sep="\t") >> ply_tdy.extract(
        "year_pair", into=["year1", "year2"], regex=r"(\d+)_\d-(\d+)_\d", convert=True
    )

    average_file_name = (
        f"output/averaged_inter_models/average_{str(Path(file).stem)}.tsv"
    )

    if Path(average_file_name).exists():
        continue

    averaged_inter_year_models = dict()
    for idx, row in tqdm.tqdm(
        inter_year_model_df.iterrows(), desc=f"inter_df {str(Path(file).stem)}"
    ):

        if (
            row["tok"],
            int(row["year1"]),
            int(row["year2"]),
        ) not in averaged_inter_year_models:
            averaged_inter_year_models[
                (row["tok"], int(row["year1"]), int(row["year2"]))
            ] = dict(global_distance=[], local_distance=[])

        averaged_inter_year_models[(row["tok"], int(row["year1"]), int(row["year2"]))][
            "global_distance"
        ].append(row["global_distance"])
        averaged_inter_year_models[(row["tok"], int(row["year1"]), int(row["year2"]))][
            "local_distance"
        ].append(row["local_distance"])

    with open(average_file_name, "w") as outfile:
        fieldnames = [
            "average_global_distance",
            "average_local_distance",
            "var_global_distance",
            "var_local_distance",
            "tok",
            "year1",
            "year2",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for tok, year1, year2 in tqdm.tqdm(
            averaged_inter_year_models, desc="summary_inter_writer"
        ):
            writer.writerow(
                {
                    "average_global_distance": np.mean(
                        averaged_inter_year_models[(tok, year1, year2)][
                            "global_distance"
                        ]
                    ),
                    "var_global_distance": np.var(
                        averaged_inter_year_models[(tok, year1, year2)][
                            "global_distance"
                        ]
                    ),
                    "average_local_distance": np.mean(
                        averaged_inter_year_models[(tok, year1, year2)][
                            "local_distance"
                        ]
                    ),
                    "var_local_distance": np.var(
                        averaged_inter_year_models[(tok, year1, year2)][
                            "local_distance"
                        ]
                    ),
                    "tok": tok,
                    "year1": year1,
                    "year2": year2,
                }
            )

# # Custom Statistic that accounts for Inter and Intra Variation

# I needed to figure out a metric to take in inter-year (between years) and intra-year(within year variation).
# Turns out population genetics developed a statistic that accounts for genetic variation between populations and with in populations (termed $Q_{st}$).
# This metric is calculated via this equation: $$Q_{st}= \frac{Variation_{between}}{Variation_{between} + 2*Variation_{within}}$$
#
# Translating this equation into my field, population is the same as a group of word2vec models trained on abstracts for a given year.
# Each "year" has it's own variation (intra) along with variation across years (inter), so the idea here is to try and capture this instability.
#
# Using the equation above as inspiration, I devise a custom equation below.
#
# First have to define the distance mapping function:
# Let distance be cosine distance: $$ distance(w_{x}, w_{y}) = cos\_dist(w_{x}, w_{y})$$ where $$ 0 \leq cos\_dist(w_{x}, w_{y}) \leq 2$$
#
# Values close to 2 signify completely opposite word contexts, while values close to 0 signify same word context.
#
# Every publication year has ten models. I took the average distance of every model combination for a given year to calculate the intra year variation for each given word.
# E.g. year 2000 has 10 choose 2 options so for every combination pair I calculated the distance above and then averaged over all years.
# For inter year I just performed the cartesian product of all models between years and then perform the same average approach above.
# Now assume each distance is averaged, we get the following equation:
#
# $$\hat{Distance} = \frac{Distance_{inter\_year(x,y)}}{Distance_{inter\_year(x,y)} + Distance_{intra\_year(x)} + Distance_{intra\_year(y)}}$$
#
# Where x and y are a particular year and $x \neq y$.
# If $x = y$ then this estimate would be 1.
#
# However, I cant use this metric for bayesian changepoint detection as this metric would be completely dominated by
# the frequency ratio metric.
# In other words the above metric is bound between 0 and 1, while the frequency ratio is bounded between 0 and infinity.
# Therefore, the change metric heavily depends on frequency to work. This is bad as there are words that have undergone a semantic change, but have yet to have a change in frequency to detect said change (e.g. increase).
#
# To account for this I'm using the following metric instead:
# $$\hat{Distance} = \frac{Distance_{inter\_year(x,y)}}{Distance_{intra\_year(x)} + Distance_{intra\_year(y)}}$$

intra_year_averaged = pd.concat(
    [
        pd.read_csv(str(file), sep="\t")
        for file in Path("output/averaged_intra_models").rglob("*.tsv")
    ]
)
intra_year_averaged.head()

tok_intra_year = dict()
for idx, row in tqdm.tqdm(intra_year_averaged.iterrows()):
    tok_intra_year[(row["tok"], row["year"])] = {
        "global": row["average_global_distance"],
        "local": row["average_local_distance"],
    }

inter_model_files = list(Path("output/averaged_inter_models").rglob("*tsv"))
unique_years = set(
    list(map(lambda x: re.search(r"(\d+)", x.stem).groups()[0], inter_model_files))
)
len(unique_years)

for year in unique_years:

    if Path(
        f"output/combined_inter_intra_distances/saved_{year}_distance.tsv"
    ).exists():
        print(f"{year} exists!")
        continue

    inter_year_models_averaged = pd.concat(
        [
            pd.read_csv(str(file), sep="\t")
            for file in filter(
                lambda x: re.search(r"(\d+)", x.stem).group(0) == year,
                Path("output/averaged_inter_models").rglob(f"*{year}*.tsv"),
            )
        ]
    )

    data = []
    already_seen = set()
    for idx, row in tqdm.tqdm(inter_year_models_averaged.iterrows()):
        # Inter year variation
        global_inter_top = row["average_global_distance"]
        local_inter_top = row["average_local_distance"]

        if (row["tok"], int(row["year1"])) not in tok_intra_year or (
            row["tok"],
            int(row["year2"]),
        ) not in tok_intra_year:
            continue

        # global intra year variation
        global_intra_bottom = (
            tok_intra_year[(row["tok"], int(row["year1"]))]["global"]
            + tok_intra_year[(row["tok"], int(row["year2"]))]["global"]
        )

        global_distance_qst = global_inter_top / (
            global_inter_top + global_intra_bottom
        )

        # local intra year variation
        # intra year variaion
        local_intra_bottom = (
            tok_intra_year[(row["tok"], int(row["year1"]))]["local"]
            + tok_intra_year[(row["tok"], int(row["year2"]))]["local"]
        )

        local_distance_qst = local_inter_top / (local_inter_top + local_intra_bottom)

        data.append(
            {
                "tok": row["tok"],
                "original_global_distance": global_inter_top,
                "global_distance_qst": global_distance_qst,
                "local_distance_qst": local_distance_qst,
                "ratio_metric": global_inter_top / global_intra_bottom,
                "year_1": row["year1"],
                "year_2": row["year2"],
            }
        )

    corrected_df = pd.DataFrame.from_records(data)
    stable_tokens = corrected_df >> ply.group_by("tok") >> ply.count()

    tokens_to_filter = stable_tokens.query(f"n=={stable_tokens.n.max()}").tok.tolist()
    corrected_df = corrected_df >> ply.query(f"tok in {tokens_to_filter}")
    corrected_df.to_csv(
        f"output/combined_inter_intra_distances/saved_{year}_distance.tsv",
        sep="\t",
        index=False,
    )
