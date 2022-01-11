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

# # Compare CUSUM against Bayesian Changepoint Detection

# This notebook is designed to compare the two detection methods. Bayesian changepoint detection always predicts right before or right after a true change point event. Reason for this is that I'm modeling changes as ratios so "spikes" in the time series cause the algorithm to have off by one errors. To circumvent this issue I tested to see if the CUSUM algorithm would have the same issue.

# +
from IPython import display
from pathlib import Path
import re

from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import pandas as pd
import plotnine as p9
import plydata as ply
import plydata.tidy as ply_tdy


# -

def plot_values(
    metric_df: pd.DataFrame,
    bayespoint_df: pd.DataFrame,
    cusumpoint_df: pd.DataFrame,
    tok: str = "the",
):
    metric_plot = (
        p9.ggplot(metric_df >> ply.query(f"tok=='{tok}'"))
        + p9.aes(x="year_pair", y="change_metric_ratio", group=0)
        + p9.geom_line()
        + p9.coord_flip()
        + p9.theme_seaborn("white")
        + p9.labs(
            x="Year Shift",
            y="Ratio",
            title=f"Frequency + Semantic Ratio for Token ('{tok}')",
        )
    )

    bayes_years_predicted = (
        bayespoint_df
        >> ply.query(f"tok == '{tok}'&changepoint_prob > 0.5")
        >> ply.pull("year_pair")
    )

    for year in bayes_years_predicted:
        metric_plot += p9.annotate(
            "point",
            x=year,
            y=(
                metric_df
                >> ply.query(f"tok=='{tok}'")
                >> ply.query(f"year_pair=='{year}'")
                >> ply.pull("change_metric_ratio")
            )[0],
            fill="purple",
            size=4,
        )

    cusum_years_predicted = (
        cusumpoint_df >> ply.query(f"tok == '{tok}'") >> ply.pull("changepoint_idx")
    )

    for year in cusum_years_predicted:
        if year in bayes_years_predicted:
            metric_plot += p9.annotate(
                "point",
                x=year,
                y=(
                    metric_df
                    >> ply.query(f"tok=='{tok}'")
                    >> ply.query(f"year_pair=='{year}'")
                    >> ply.pull("change_metric_ratio")
                )[0],
                fill="green",
                size=4,
            )
        else:
            metric_plot += p9.annotate(
                "point",
                x=year,
                y=(
                    metric_df
                    >> ply.query(f"tok=='{tok}'")
                    >> ply.query(f"year_pair=='{year}'")
                    >> ply.pull("change_metric_ratio")
                )[0],
                fill="red",
                size=4,
            )

    return metric_plot


# ## Load the data

change_metric_df = pd.read_csv("output/change_metric_abstracts.tsv", sep="\t")
change_metric_df >> ply.slice_rows(10)

bayes_point_df = pd.read_csv("output/bayesian_changepoint_data_abstracts.tsv", sep="\t")
bayes_point_df >> ply.slice_rows(10)

cusum_point_df = pd.read_csv("output/cusum_changepoint_abstracts.tsv", sep="\t")
cusum_point_df >> ply.slice_rows(10)

# # Selected Tokens to Visualize Changepoints

# Let's take a look a the predictions from both methods. The CUSUM method is in red while the bayesian changepoint detection algorithm is in green. The purple dots are algorithm prediction agreements.

# ## Pandemic

plot_values(change_metric_df, bayes_point_df, cusum_point_df, tok="pandemic")

# ## Abcc6

plot_values(change_metric_df, bayes_point_df, cusum_point_df, tok="abcc6")

# ## asthmatics

plot_values(change_metric_df, bayes_point_df, cusum_point_df, tok="asthmatics")

# ## 2005

plot_values(change_metric_df, bayes_point_df, cusum_point_df, tok="2005")

# # Take Home Points

# 1. Switch to CUSUM as it can accurately detect these spikes rather than a Bayesian method.
# 2. Raises the pitfall that ratios might not be the best way to model these changes; however, this is the only solution I can devise.
