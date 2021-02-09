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

# +
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9
from sklearn.metrics import (
    auc,
    roc_curve,
    precision_recall_curve,
)
# -

# ## Load and Parse the performance data

decade_df_dict = {}
years = [1971, 1981, 1991, 2001, 2011]

# iterate through results from 1971-2020 by decade
for year in years:
    scores_df = pd.read_csv(
        Path(f"outputs/decades/similarity_scores_{str(year)}-{str(year+9)}.tsv"),
        sep="\t",
    )

    fp, tp, _ = roc_curve(
        scores_df["class"].values.tolist(), scores_df.score.values.tolist()
    )

    precision, recall, _ = precision_recall_curve(
        scores_df["class"].values.tolist(), scores_df.score.values.tolist()
    )

    decade_df_dict[f"{str(year)}-{str(year+9)}"] = pd.DataFrame(
        {"fp": fp, "tp": tp, "precision": precision, "recall": recall}
    )

# ## Plot Performance

year = 2001

g = (
    p9.ggplot(decade_df_dict[f"{str(year)}-{str(year+9)}"], p9.aes(x="fp", y="tp"))
    + p9.geom_point()
    + p9.geom_line(p9.aes(x=[0, 1], y=[0, 1]), linetype="dashed")
)
print(g)

g = (
    p9.ggplot(
        decade_df_dict[f"{str(year)}-{str(year+9)}"],
        p9.aes(x="recall", y="precision"),
    )
    + p9.geom_point()
)
print(g)
