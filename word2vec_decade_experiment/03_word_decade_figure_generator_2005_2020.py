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

# # Generate Figures for Word2Vec Year Models

# The goal of this notebook is to observe how words are shifting through time since 2005. The year 2005 was selected as this is the only year the word 'CRISPR' appears in the Word2Vec models. This highlights the catch with Word2Vec models as they require words to appear at a given frequency within abstracts to be captured by the model.

# +
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import plotnine as p9
import tqdm

from msp_tsne import MultiscaleParametricTSNE
import tensorflow as tf
from tensorflow import keras

from biovectors_modules.plot_helper import (
    plot_token_timeline,
    plot_local_global_distances,
    plot_wordcloud_neighbors,
    plot_wordcloud_neighbors_gif,
)
from biovectors_modules.word2vec_analysis_helper import (
    generate_timeline,
    get_neighbors,
    project_token_timeline,
    window,
)
# -

# # Load Models to Observe Changes

aligned_models = pickle.load(open("output/aligned_word_vectors_2005_2020.pkl", "rb"))

year_comparison_dict = {
    "_".join(comparison_file.stem.split("_")[0:2]): (
        pd.read_csv(str(comparison_file), sep="\t")
    )
    for comparison_file in (list(Path("output/year_distances_2005_2020").rglob("*tsv")))
}
list(year_comparison_dict.keys())[0:3]

year_comparison_dict["2005_2006"].sort_values("global_dist")

# ## Train TSNE Model to Project Time Shifts into Two Dimensional Space

# The goal here is to train a TSNE model that projects all words from 2005 to 2020 into a two dimensional space. Allows one to visually track how a word vector is shifting through time.

word_models_stacked = np.vstack(list(aligned_models.values())[:-1])
file_name = "output/2005_2020_model.h5"

if not Path(file_name).exists():
    tf.random.set_random_seed(100)
    np.random.seed(100)
    model = MultiscaleParametricTSNE(n_iter=300)
    model.fit(word_models_stacked)
    keras.models.save_model(model._model, file_name)
else:
    model = MultiscaleParametricTSNE(n_iter=300)
    model._build_model(300, 2)
    model._model.load_weights(file_name)

# # Visualize Words Shifting through Time

# Global distance measures how far a word has moved within semantic space. This measure captures how words change globally across time periods. The greater the distance the more semantic change a word has been subjected towards.

# ## CRISPR

token_timeline_df = generate_timeline(year_comparison_dict, "crispr")
token_timeline_df.head()

token_timeline_low_dim_df = project_token_timeline(
    "crispr", aligned_models, model, neighbors=25
)
token_timeline_low_dim_df.head()

global_distance, local_distance = plot_local_global_distances(
    token_timeline_df, token="crispr"
)

global_distance

local_distance

g = plot_token_timeline(token_timeline_low_dim_df)
g.save("output/timeline_figures/crispr_time_plot.png")
print(g)

plot_wordcloud_neighbors_gif(
    token_timeline_low_dim_df,
    plot_filename="output/wordcloud_plots_2005_2020/crispr.gif",
)

# ![crispr gif here](output/wordcloud_plots_2005_2020/crispr.gif)

plot_wordcloud_neighbors(
    token_timeline_low_dim_df,
    plot_filename="output/wordcloud_plots_2005_2020/crispr.png",
)

# # Conclusions - Take Home Point(s)

# 1. CRISPR has a nice transition from plant biology to genome editing. Word cloud neighbors does wonders in observing this transition.
