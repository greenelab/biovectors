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

# # Statistics to Correct for Alignment Issues

# +
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import plotnine as p9
from plydata import (
    query,
    group_by,
    select,
    if_else,
    summarize,
    pull,
    define,
    sample_n,
    do,
    arrange,
)
from plydata.tidy import extract

from biovectors_modules.plot_helper import plot_local_global_distances
# -

# # Load Data

distance_file = "output/all_year_distances.tsv.xz"
if not Path(distance_file).exists():
    data_df = []
    for file in Path("output").rglob("*part*.tsv"):
        data_df.append(pd.read_csv(file, sep="\t"))
    data_df = pd.concat(data_df).sort_values("token")
    data_df.to_csv(distance_file, sep="\t", index=False, compression="xz")
else:
    data_df = pd.read_csv(distance_file, sep="\t")
data_df.sample(100, random_state=100)

tokens = (
    data_df
    >> query("year_pair.str.contains('2000-2001')")
    >> sample_n(1000, random_state=100)
    >> pull("token")
)
tokens

# # Calculate Z scores from the Data

z_score_df = (
    data_df
    >> query("token in @tokens")
    >> group_by("year_pair")
    >> summarize(
        global_mean="np.mean(global_dist)",
        global_std="np.std(global_dist)",
        local_mean="np.mean(local_dist)",
        local_std="np.std(local_dist)",
    )
)
z_score_df

years = [
    f"{pair[0]}-{pair[1]}"
    for pair in itertools.combinations(range(2000, 2021), 2)
    if pair[1] - pair[0] == 1
]

g = (
    p9.ggplot(
        data_df >> query("token in @tokens&year_pair in @years"),
        p9.aes(x="factor(year_pair)", y="global_dist"),
    )
    + p9.geom_boxplot()
    + p9.coord_flip()
)
print(g)

g = (
    p9.ggplot(
        data_df >> query("token in @tokens&year_pair in @years"),
        p9.aes(x="factor(year_pair)", y="local_dist"),
    )
    + p9.geom_boxplot()
    + p9.coord_flip()
)
print(g)

year_pair_mapper = z_score_df.set_index("year_pair").to_dict("index")

plot_df = (
    data_df.append(pd.read_csv("caov3_fixed.tsv", sep="\t", index_col=0)).append(
        pd.read_csv("crispr_fixed.tsv", sep="\t", index_col=0)
    )
    >> define(
        z_global_dist=lambda x: (
            x.global_dist
            - x.year_pair.apply(lambda pair: year_pair_mapper[pair]["global_mean"])
        )
        / x.year_pair.apply(lambda pair: year_pair_mapper[pair]["global_std"]),
        z_local_dist=lambda x: (
            x.global_dist
            - x.year_pair.apply(lambda pair: year_pair_mapper[pair]["local_mean"])
        )
        / x.year_pair.apply(lambda pair: year_pair_mapper[pair]["local_std"]),
    )
    >> extract("year_pair", into=["year_origin", "year_compared"], regex=r"(\d+)-(\d+)")
)
plot_df

# # Plot Token Distances

figure_dir = Path("output/distance_heatmaps")

# ## Are Token

global_plot, local_plot, z_global_plot, z_local_plot = plot_local_global_distances(
    plot_df, "are"
)

global_plot.save(f"{figure_dir}/are_global_heatmap.png", dpi=150)
global_plot

local_plot.save(f"{figure_dir}/are_local_heatmap.png", dpi=150)
local_plot

z_global_plot.save(f"{figure_dir}/are_z_global_heatmap.png", dpi=150)
z_global_plot

z_local_plot.save(f"{figure_dir}/are_z_local_heatmap.png", dpi=150)
z_local_plot

# ## Interleukin-18

global_plot, local_plot, z_global_plot, z_local_plot = plot_local_global_distances(
    plot_df, "interleukin-18"
)

global_plot.save(f"{figure_dir}/interleukin-18_global_heatmap.png", dpi=150)
global_plot

local_plot.save(f"{figure_dir}/interleukin-18_local_heatmap.png", dpi=150)
local_plot

z_global_plot.save(f"{figure_dir}/interleukin-18_z_global_heatmap.png", dpi=150)
z_global_plot

z_local_plot.save(f"{figure_dir}/interleukin-18_z_local_heatmap.png", dpi=150)
z_local_plot

# ## Crispr

global_plot, local_plot, z_global_plot, z_local_plot = plot_local_global_distances(
    plot_df, "crispr"
)

global_plot.save(f"{figure_dir}/crispr_global_heatmap.png", dpi=150)
global_plot

local_plot.save(f"{figure_dir}/crispr_local_heatmap.png", dpi=150)
local_plot

z_global_plot.save(f"{figure_dir}/crispr_z_global_heatmap.png", dpi=150)
z_global_plot

z_local_plot.save(f"{figure_dir}/crispr_z_local_heatmap.png", dpi=150)
z_local_plot

# ## Caov3

global_plot, local_plot, z_global_plot, z_local_plot = plot_local_global_distances(
    plot_df, "caov3"
)

global_plot.save(f"{figure_dir}/caov3_global_heatmap.png", dpi=150)
global_plot

local_plot.save(f"{figure_dir}/caov3_local_heatmap.png", dpi=150)
local_plot

z_global_plot.save(f"{figure_dir}/caov3_z_global_heatmap.png", dpi=150)
z_global_plot

z_local_plot.save(f"{figure_dir}/caov3_z_local_heatmap.png", dpi=150)
z_local_plot

# # Conclusion

# 1. Stop words are a great negative control to show how words shouldn't change through time.
# 2. CRISPR is a great positive shows how the vector is changing through time with the change being stable after 2009.
# 3. Interleukin 18 seems to have a change in the year 2005, but the open problem here is that 2005 may not be aligned correctly which results in this "seemingly" positive result.
# 4. Next step here is to see if I can account for this alignment issue when working on this project.
