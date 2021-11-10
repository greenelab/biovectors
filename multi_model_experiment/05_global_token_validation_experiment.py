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

# Early year models are unstable there isn't a whole lot of data to appropriately train word2vec model. [04_novel_distance_calculations](04_novel_distance_calculations.ipynb) notebook was designed to create a metric that can account for this problem.
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

# ## load the years

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

# ## Filter data into 1 year consecutive distances

full_token_set_df = pd.concat(
    [
        year_distance_map[year]
        >> ply.query(f"tok in {token_filter_list}")
        >> ply.query("year_2-year_1 == 1")
        for year in tqdm.tqdm(year_distance_map)
    ]
)
print(full_token_set_df.shape)
full_token_set_df.head()

# # Does distance increase across the years?

original_df = (
    full_token_set_df
    >> ply.arrange("tok")
    >> ply.group_by("year_2")
    >> ply.define(avg_global_distance="mean(original_global_distance)")
    >> ply.select("avg_global_distance", "year_1")
    >> ply.distinct()
    >> ply.ungroup()
    >> ply_tdy.unite("year_pair", "year_1", "year_2")
)
original_df >> ply.slice_rows(10)

qst_df = (
    full_token_set_df
    >> ply.arrange("tok")
    >> ply.group_by("year_2")
    >> ply.define(avg_global_distance_qst="mean(global_distance_qst)")
    >> ply.select("avg_global_distance_qst", "year_1")
    >> ply.distinct()
    >> ply.ungroup()
    >> ply_tdy.unite("year_pair", "year_1", "year_2")
)
qst_df >> ply.slice_rows(10)

times_qst_df = (
    full_token_set_df
    >> ply.arrange("tok")
    >> ply.group_by("year_2")
    >> ply.define(avg_global_times_distance_qst="mean(global_times_distance_qst)")
    >> ply.select("avg_global_times_distance_qst", "year_1")
    >> ply.distinct()
    >> ply.ungroup()
    >> ply_tdy.unite("year_pair", "year_1", "year_2")
)
times_qst_df >> ply.slice_rows(10)

global_qst_comparison = (
    original_df
    >> ply.inner_join(qst_df)
    >> ply.inner_join(times_qst_df)
    >> ply_tdy.gather(
        "metric",
        "distance_value",
        ply.select(
            "avg_global_distance_qst",
            "avg_global_times_distance_qst",
            "avg_global_distance",
        ),
    )
)
global_qst_comparison.year_pair = pd.Categorical(
    global_qst_comparison.year_pair.tolist()
)
global_qst_comparison.head(10)

(
    p9.ggplot(
        global_qst_comparison,
        p9.aes(x="year_pair", y="distance_value", color="metric", group="metric"),
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.coord_flip()
    + p9.theme_seaborn(style="white")
    + p9.scale_color_brewer("qual", palette=2)
    + p9.labs(y="Distance", x="Year")
)

middle_map = dict(
    global_qst_comparison
    >> ply.query("year_pair=='2010_2011'")
    >> ply.pull(["metric", "distance_value"])
)

(
    global_qst_comparison
    >> ply.arrange("year_pair")
    >> ply.define(
        percent_diff=ply.expressions.case_when(
            {
                'metric=="avg_global_distance_qst"': f'distance_value/{middle_map["avg_global_distance_qst"]} - 1',
                'metric=="avg_global_distance"': f'distance_value/{middle_map["avg_global_distance"]} - 1',
                'metric=="avg_global_times_distance_qst"': f'distance_value/{middle_map["avg_global_times_distance_qst"]} - 1',
            }
        )
    )
    >> (
        p9.ggplot(
            p9.aes(x="year_pair", y="percent_diff", group="metric", color="metric")
        )
        + p9.geom_point()
        + p9.geom_line()
        + p9.coord_flip()
        + p9.theme_seaborn(style="white")
        + p9.labs(title="Percent Difference Relative to 2010-2011")
        + p9.scale_color_brewer("qual", palette=2)
    )
)

(
    global_qst_comparison
    >> ply.arrange("year_pair")
    >> ply.define(
        percent_diff=ply.expressions.case_when(
            {
                'metric=="avg_global_distance_qst"': f'distance_value/{middle_map["avg_global_distance_qst"]} - 1',
                'metric=="avg_global_distance"': f'distance_value/{middle_map["avg_global_distance"]} - 1',
                'metric=="avg_global_times_distance_qst"': f'distance_value/{middle_map["avg_global_times_distance_qst"]} - 1',
            }
        )
    )
    >> ply.define(abs_percent_diff="abs(percent_diff)")
    >> (
        p9.ggplot(
            p9.aes(x="year_pair", y="abs_percent_diff", group="metric", color="metric")
        )
        + p9.geom_point()
        + p9.geom_line()
        + p9.coord_flip()
        + p9.theme_seaborn(style="white")
        + p9.labs(title="Absolute Percent Difference Relative to 2010-2011")
        + p9.scale_color_brewer("qual", palette=2)
    )
)

# # Does distance increase across the years when focusing on a single year?

# ## 2000

full_token_set_df = pd.concat(
    [
        year_distance_map[year]
        >> ply.query(f"tok in {token_filter_list}")
        >> ply.query("year_1 == 2000")
        for year in tqdm.tqdm(year_distance_map)
    ]
)
print(full_token_set_df.shape)
full_token_set_df.head()

original_df = (
    full_token_set_df
    >> ply.arrange("tok")
    >> ply.group_by("year_2")
    >> ply.define(avg_global_distance="mean(original_global_distance)")
    >> ply.select("avg_global_distance", "year_1")
    >> ply.distinct()
    >> ply.ungroup()
    >> ply_tdy.unite("year_pair", "year_1", "year_2")
)
original_df >> ply.slice_rows(10)

qst_df = (
    full_token_set_df
    >> ply.arrange("tok")
    >> ply.group_by("year_2")
    >> ply.define(avg_global_distance_qst="mean(global_distance_qst)")
    >> ply.select("avg_global_distance_qst", "year_1")
    >> ply.distinct()
    >> ply.ungroup()
    >> ply_tdy.unite("year_pair", "year_1", "year_2")
)
qst_df >> ply.slice_rows(10)

times_qst_df = (
    full_token_set_df
    >> ply.arrange("tok")
    >> ply.group_by("year_2")
    >> ply.define(avg_global_times_distance_qst="mean(global_times_distance_qst)")
    >> ply.select("avg_global_times_distance_qst", "year_1")
    >> ply.distinct()
    >> ply.ungroup()
    >> ply_tdy.unite("year_pair", "year_1", "year_2")
)
times_qst_df >> ply.slice_rows(10)

global_qst_comparison = (
    original_df
    >> ply.inner_join(qst_df)
    >> ply.inner_join(times_qst_df)
    >> ply_tdy.gather(
        "metric",
        "distance_value",
        ply.select(
            "avg_global_distance_qst",
            "avg_global_times_distance_qst",
            "avg_global_distance",
        ),
    )
)
global_qst_comparison.year_pair = pd.Categorical(
    global_qst_comparison.year_pair.tolist()
)
global_qst_comparison.head(10)

(
    p9.ggplot(
        global_qst_comparison,
        p9.aes(x="year_pair", y="distance_value", color="metric", group="metric"),
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.coord_flip()
    + p9.theme_seaborn(style="white")
    + p9.scale_color_brewer("qual", palette=2)
    + p9.labs(y="Distance", x="Year")
)

middle_map = dict(
    global_qst_comparison
    >> ply.query("year_pair=='2000_2010'")
    >> ply.pull(["metric", "distance_value"])
)

(
    global_qst_comparison
    >> ply.arrange("year_pair")
    >> ply.define(
        percent_diff=ply.expressions.case_when(
            {
                'metric=="avg_global_distance_qst"': f'abs(distance_value/{middle_map["avg_global_distance_qst"]} - 1)',
                'metric=="avg_global_distance"': f'abs(distance_value/{middle_map["avg_global_distance"]} - 1)',
                'metric=="avg_global_times_distance_qst"': f'abs(distance_value/{middle_map["avg_global_times_distance_qst"]} - 1)',
            }
        )
    )
    >> (
        p9.ggplot(
            p9.aes(x="year_pair", y="percent_diff", group="metric", color="metric")
        )
        + p9.geom_point()
        + p9.geom_line()
        + p9.coord_flip()
        + p9.theme_seaborn(style="white")
        + p9.labs(title="Absolute Percent Difference Relative to 2010-2011")
        + p9.scale_color_brewer("qual", palette=2)
    )
)

# ## 2001

full_token_set_df = pd.concat(
    [
        year_distance_map[year]
        >> ply.query(f"tok in {token_filter_list}")
        >> ply.query("year_1 == 2001")
        for year in tqdm.tqdm(year_distance_map)
    ]
)
print(full_token_set_df.shape)
full_token_set_df.head()

original_df = (
    full_token_set_df
    >> ply.arrange("tok")
    >> ply.group_by("year_2")
    >> ply.define(avg_global_distance="mean(original_global_distance)")
    >> ply.select("avg_global_distance", "year_1")
    >> ply.distinct()
    >> ply.ungroup()
    >> ply_tdy.unite("year_pair", "year_1", "year_2")
)
original_df >> ply.slice_rows(10)

qst_df = (
    full_token_set_df
    >> ply.arrange("tok")
    >> ply.group_by("year_2")
    >> ply.define(avg_global_distance_qst="mean(global_distance_qst)")
    >> ply.select("avg_global_distance_qst", "year_1")
    >> ply.distinct()
    >> ply.ungroup()
    >> ply_tdy.unite("year_pair", "year_1", "year_2")
)
qst_df >> ply.slice_rows(10)

times_qst_df = (
    full_token_set_df
    >> ply.arrange("tok")
    >> ply.group_by("year_2")
    >> ply.define(avg_global_times_distance_qst="mean(global_times_distance_qst)")
    >> ply.select("avg_global_times_distance_qst", "year_1")
    >> ply.distinct()
    >> ply.ungroup()
    >> ply_tdy.unite("year_pair", "year_1", "year_2")
)
times_qst_df >> ply.slice_rows(10)

global_qst_comparison = (
    original_df
    >> ply.inner_join(qst_df)
    >> ply.inner_join(times_qst_df)
    >> ply_tdy.gather(
        "metric",
        "distance_value",
        ply.select(
            "avg_global_distance_qst",
            "avg_global_times_distance_qst",
            "avg_global_distance",
        ),
    )
)
global_qst_comparison.year_pair = pd.Categorical(
    global_qst_comparison.year_pair.tolist()
)
global_qst_comparison.head(10)

(
    p9.ggplot(
        global_qst_comparison,
        p9.aes(x="year_pair", y="distance_value", color="metric", group="metric"),
    )
    + p9.geom_point()
    + p9.geom_line()
    + p9.coord_flip()
    + p9.theme_seaborn(style="white")
    + p9.scale_color_brewer("qual", palette=2)
    + p9.labs(y="Distance", x="Year")
)

middle_map = dict(
    global_qst_comparison
    >> ply.query("year_pair=='2001_2010'")
    >> ply.pull(["metric", "distance_value"])
)

(
    global_qst_comparison
    >> ply.arrange("year_pair")
    >> ply.define(
        percent_diff=ply.expressions.case_when(
            {
                'metric=="avg_global_distance_qst"': f'abs(distance_value/{middle_map["avg_global_distance_qst"]} - 1)',
                'metric=="avg_global_distance"': f'abs(distance_value/{middle_map["avg_global_distance"]} - 1)',
                'metric=="avg_global_times_distance_qst"': f'abs(distance_value/{middle_map["avg_global_times_distance_qst"]} - 1)',
            }
        )
    )
    >> (
        p9.ggplot(
            p9.aes(x="year_pair", y="percent_diff", group="metric", color="metric")
        )
        + p9.geom_point()
        + p9.geom_line()
        + p9.coord_flip()
        + p9.theme_seaborn(style="white")
        + p9.labs(title="Absolute Percent Difference Relative to 2010-2011")
        + p9.scale_color_brewer("qual", palette=2)
    )
)

# Take home Points:
# 1. The metric that appears to be most stable is the qst metric: $\hat{Distance} = \frac{Distance_{inter\_year(x,y)}}{Distance_{inter\_year(x,y)} + Distance_{intra\_year(x)} + Distance_{intra\_year(y)}}$. Reason for this claim is that the green line in the following graphics above doesn't appear to have drastic spikes. This suggests year instability is corrected by using this metric.
# 2. When comparing one year to the rest the distance does change over time which is exactly what we were expecting to observe.
