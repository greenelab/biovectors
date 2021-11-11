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

# # Does New Metric Solve the Early Year Instability Problem?

# Early year models are unstable as there isn't a whole lot of data to appropriately train a word2vec model. [04_novel_distance_calculations](04_novel_distance_calculations.ipynb) notebook was designed to create a metric that can account for this problem.
# This notebook is designed to test if the newly constructed metric actually fixes the problem via a global comparison using all tokens present in 2000 through 2020.
#
# *Note*: compare the global qst metric to one of the earlier year models. The comparison should be similar to the one I did on a global scale.
# Goal here is to show that this metric works better to handle the model instability that occurs.

# +
from pathlib import Path
import re

import pandas as pd
import plotnine as p9
import plydata as ply
import plydata.tidy as ply_tdy
import tqdm
# -

# # Load distances for all Year Pairs

# ## Load the years for metric fix

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
        for year in tqdm.tqdm(year_distance_map)
    ]
)
print(full_token_set_df.shape)
full_token_set_df.head()

# ## Load individual Models for a couple years to see the change

all_inter_year_models = list(Path("output/inter_models").rglob("*tsv"))

regex_pattern = r"\d+_0-\d+_0"
all_inter_models_df = pd.concat(
    [
        pd.read_csv(str(file), sep="\t")
        >> ply.query(f"year_pair.str.contains({regex_pattern})")
        for file in all_inter_year_models
    ]
)
all_inter_models_df >> ply.slice_rows(15)

middle_estimate = (
    all_inter_models_df
    >> ply.query("year_pair=='2010_0-2011_0'")
    >> ply.pull("global_distance")
).mean()
middle_estimate

regex_pattern = r"(\d+)_0-(\d+)_0"
percent_diff_one_year = (
    all_inter_models_df
    >> ply_tdy.extract(
        "year_pair", into=["year_1", "year_2"], regex=regex_pattern, convert=True
    )
    >> ply.group_by("year_1", "year_2")
    >> ply.query("year_2-year_1==1")
    >> ply.define(pct_diff="mean(global_distance)/middle_estimate - 1")
    >> ply.select("year_1", "year_2", "pct_diff")
    >> ply.ungroup()
    >> ply.distinct()
    >> ply_tdy.unite("year_pair", "year_1", "year_2")
)
percent_diff_one_year

# # Does distance increase across the years?

all_distance_df = (
    full_token_set_df
    >> ply.arrange("tok")
    >> ply.group_by("year_2")
    >> ply.define(
        avg_global_distance="mean(original_global_distance)",
        avg_global_distance_qst="mean(global_distance_qst)",
        avg_ratio_metric="mean(ratio_metric)",
    )
    >> ply.select(
        "avg_global_distance", "avg_global_distance_qst", "avg_ratio_metric", "year_1"
    )
    >> ply.distinct()
    >> ply.ungroup()
    >> ply.query("year_2-year_1==1")
    >> ply_tdy.unite("year_pair", "year_1", "year_2")
)
all_distance_df >> ply.slice_rows(15)

global_qst_comparison = all_distance_df >> ply_tdy.gather(
    "metric",
    "distance_value",
    ply.select(
        "avg_global_distance_qst",
        "avg_global_distance",
        "avg_ratio_metric",
    ),
)
global_qst_comparison.year_pair = pd.Categorical(
    global_qst_comparison.year_pair.tolist()
)
global_qst_comparison.head(10)

middle_map = dict(
    global_qst_comparison
    >> ply.query("year_pair=='2010_2011'")
    >> ply.pull(["metric", "distance_value"])
)

pct_diff_df = (
    global_qst_comparison
    >> ply.arrange("year_pair")
    >> ply.define(
        pct_diff=ply.expressions.case_when(
            {
                'metric=="avg_global_distance_qst"': f'distance_value/{middle_map["avg_global_distance_qst"]} - 1',
                'metric=="avg_global_distance"': f'distance_value/{middle_map["avg_global_distance"]} - 1',
                'metric=="avg_ratio_metric"': f'distance_value/{middle_map["avg_ratio_metric"]} - 1',
            }
        )
    )
)
pct_diff_df

(
    p9.ggplot(
        pct_diff_df
        >> ply.select("-distance_value")
        >> ply.call(
            ".append", percent_diff_one_year >> ply.define(metric='"single_model"')
        ),
        p9.aes(x="year_pair", y="abs(pct_diff)", group="metric", color="metric"),
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.coord_flip()
    + p9.theme_seaborn(style="white")
    + p9.labs(title="Percent Difference Relative to 2010-2011")
    + p9.scale_color_brewer("qual", palette=2)
)

g = (
    p9.ggplot(
        pct_diff_df
        >> ply.select("-distance_value")
        >> ply.call(
            ".append", percent_diff_one_year >> ply.define(metric='"single_model"')
        )
        >> ply.query("metric=='single_model' or metric=='avg_ratio_metric'")
        >> ply.define(
            metric=ply.expressions.case_when(
                {'metric=="avg_ratio_metric"': '"correction_model"', True: "metric"}
            )
        ),
        p9.aes(x="year_pair", y="abs(pct_diff)", group="metric", color="metric"),
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.coord_flip()
    + p9.theme_seaborn(style="white")
    + p9.labs(title="Percent Difference Relative to 2010-2011")
    + p9.scale_color_brewer("qual", palette=2)
    + p9.labs(x="Year Shifts", y="Absolute Percent Difference")
)
g.save("output/figures/early_year_correction_comparison.svg")
g.save("output/figures/early_year_correction_comparison.png", dpi=500)
print(g)

# Take home Points:
# 1. All metrics (using more than one model, using the qst metric, using qst * original distance, using the ratio metric) in the first figure shows seem to improve early year bias. Not a drastic spike in absolute percent difference.
# 2. Still resorting to using the ratio metric as mentioned in previous notebook.
# 3. The last figure highlights a need for a correction metric as well. I filtered out the other metric ideas and only used the ratio metric (green) compared to the traditional metric (orange).
