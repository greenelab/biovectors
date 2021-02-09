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

# # Analyze Word2Vec's Performance

# +
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
# -

# ## Load and Parse the performance data

scores_df = pd.read_csv(Path("output/similarity_scores.tsv"), sep="\t")
print(scores_df.shape)
scores_df.head()

(
    scores_df.sort_values("score", ascending=False).to_csv(
        "output/sorted_similarity_scores.tsv", sep="\t", index=False
    )
)

fp, tp, _ = roc_curve(
    scores_df["class"].values.tolist(), scores_df.score.values.tolist()
)

precision, recall, _ = precision_recall_curve(
    scores_df["class"].values.tolist(), scores_df.score.values.tolist()
)

avg_precision = average_precision_score(
    scores_df["class"].values.tolist(), scores_df.score.values.tolist()
)

performance_df = pd.DataFrame(
    {"fp": fp, "tp": tp, "precision": precision, "recall": recall}
)
performance_df.head()

# ## Plot the Performance

g = (
    p9.ggplot(performance_df, p9.aes(x="fp", y="tp"))
    + p9.geom_point()
    + p9.geom_line(p9.aes(x=[0, 1], y=[0, 1]), linetype="dashed")
)
print(g)

g = (
    p9.ggplot(
        performance_df,
        p9.aes(x="recall", y="precision", label=f"Word2Vec {avg_precision:0.2f}"),
    )
    + p9.geom_point()
    + p9.geom_hline(p9.aes(yintercept=scores_df["class"].mean()), linetype="dashed")
)
print(g)
