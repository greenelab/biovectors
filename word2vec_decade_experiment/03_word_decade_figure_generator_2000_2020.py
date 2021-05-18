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

# The goal of this notebook is to observe how words are shifting through time since 2000. The caveat here is that words have to been present within all time periods in order to be present for this task. Ideally words to be examined so far are: 'expression', 'are', 'interleukin-18', '95%ci', and 'p53'.

# +
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import plotnine as p9
import tqdm
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP

from biovectors_modules.plot_helper import (
    plot_local_global_distances,
    plot_token_timeline,
    plot_wordcloud_neighbors,
    plot_wordcloud_neighbors_gif,
)
from biovectors_modules.word2vec_analysis_helper import (
    generate_timeline,
    get_neighbors,
    project_token_timeline,
)
# -

# # Load Models to Observe Changes

# ## Load Aligned Word Vectors

aligned_models = pickle.load(open("output/aligned_word_vectors_2000_2020.pkl", "rb"))

year_comparison_dict = {
    "_".join(comparison_file.stem.split("_")[0:2]): (
        pd.read_csv(str(comparison_file), sep="\t")
    )
    for comparison_file in (list(Path("output/year_distances_2000_2020").rglob("*tsv")))
}
list(year_comparison_dict.keys())[0:3]

year_comparison_dict["2000_2005"].sort_values("global_dist", ascending=False)

# ## Train UMAP Model to Project Time Shifts into Two Dimensional Space

# The goal here is to train a TSNE model that projects all words from 2000 to 2020 into a two dimensional space. Allows one to visually track how a word vector is shifting through time.

origin_df = aligned_models["2000"]
word_vectors = list(
    map(
        lambda x: x.query(f"token in {origin_df.token.tolist()}")
        .sort_values("token")
        .set_index("token")
        .values,
        aligned_models.values(),
    )
)

word_models_stacked = np.vstack(word_vectors)
file_name = "output/2000_2020_umap_model"

if not Path(file_name).exists():
    Path(file_name).mkdir(parents=True)
    model = ParametricUMAP(
        verbose=True, metric="cosine", random_state=100, low_memory=True, n_neighbors=25
    )
    embedding = model.fit_transform(word_models_stacked)
    model.save(file_name)
else:
    model = load_ParametricUMAP(file_name)
model.verbose = False

# # Observe Diachronic Vector Changes

# Global distance measures how far a word has moved within semantic space. This measure captures how words change globally across time periods. The greater the distance the more semantic change a word has been subjected towards.
# The word clouds depict the neighbors for each word vector. The size for each token  appears to be different but size doesn't matter in this case. Each word has equal weighting.

wordcloud_folder = Path("output/wordcloud_plots_2000_2020")
timeline_folder = Path("output/timeline_figures")

# ## Are

token_timeline_df = generate_timeline(year_comparison_dict, "are")
token_timeline_df.head()

token_timeline_low_dim_df = project_token_timeline(
    "are", aligned_models, model, neighbors=25
)
token_timeline_low_dim_df.query("label=='main'")

global_distance, local_distance = plot_local_global_distances(token_timeline_df, "are")

global_distance

local_distance

g = plot_token_timeline(token_timeline_low_dim_df)
g.save(f"{str(timeline_folder)}/are_time_plot.png")
print(g)

plot_wordcloud_neighbors_gif(
    token_timeline_low_dim_df, plot_filename=f"{str(wordcloud_folder)}/are.gif"
)

# ![are gif here](output/wordcloud_plots_2000_2020/are.gif)

plot_wordcloud_neighbors(
    token_timeline_low_dim_df, plot_filename=f"{str(wordcloud_folder)}/are.png"
)

# ## Expression

token_timeline_df = generate_timeline(year_comparison_dict, "expression")
token_timeline_df.head()

token_timeline_low_dim_df = project_token_timeline(
    "expression", aligned_models, model, neighbors=25
)
token_timeline_low_dim_df.query("label=='main'")

global_distance, local_distance = plot_local_global_distances(
    token_timeline_df, "expression"
)

global_distance

local_distance

g = plot_token_timeline(token_timeline_low_dim_df)
g.save(f"{str(timeline_folder)}/expression_time_plot.png")
print(g)

plot_wordcloud_neighbors_gif(
    token_timeline_low_dim_df,
    plot_filename=f"{str(wordcloud_folder)}/expression.gif",
)

# ![expression gif here](output/wordcloud_plots_2000_2020/expression.gif)

plot_wordcloud_neighbors(
    token_timeline_low_dim_df,
    plot_filename=f"{str(wordcloud_folder)}/expression.png",
)

# ## 95%ci

token_timeline_df = generate_timeline(year_comparison_dict, "95%ci")
token_timeline_df.head()

token_timeline_low_dim_df = project_token_timeline(
    "95%ci", aligned_models, model, neighbors=25
)
token_timeline_low_dim_df.query("label=='main'")

global_distance, local_distance = plot_local_global_distances(
    token_timeline_df, token="95%ci"
)

global_distance

local_distance

g = plot_token_timeline(token_timeline_low_dim_df)
g.save(f"{str(timeline_folder)}/95_ci_time_plot.png")
print(g)

plot_wordcloud_neighbors_gif(
    token_timeline_low_dim_df,
    plot_filename=f"{str(wordcloud_folder)}/95%ci.gif",
)

# ![95%ci gif here](output/wordcloud_plots_2000_2020/95%ci.gif)

plot_wordcloud_neighbors(
    token_timeline_low_dim_df,
    plot_filename=f"{str(wordcloud_folder)}/95%ci.png",
)

# ## interleukin-18

token_timeline_df = generate_timeline(year_comparison_dict, "interleukin-18")
token_timeline_df.head()

token_timeline_low_dim_df = project_token_timeline(
    "interleukin-18", aligned_models, model, neighbors=25
)
token_timeline_low_dim_df.query("label=='main'")

global_distance, local_distance = plot_local_global_distances(
    token_timeline_df, token="interleukin-18"
)

global_distance

local_distance

g = plot_token_timeline(token_timeline_low_dim_df)
g.save(f"{str(timeline_folder)}/interleukin18_time_plot.png")
print(g)

plot_wordcloud_neighbors_gif(
    token_timeline_low_dim_df,
    plot_filename=f"{str(wordcloud_folder)}/interleukin-18.gif",
)

# ![interleukin-18 gif here](output/wordcloud_plots_2000_2020/interleukin-18.gif)

plot_wordcloud_neighbors(
    token_timeline_low_dim_df,
    plot_filename=f"{str(wordcloud_folder)}/interleukin_18.png",
)

# ## p53

token_timeline_df = generate_timeline(year_comparison_dict, "p53")
token_timeline_df.head()

token_timeline_low_dim_df = project_token_timeline(
    "p53", aligned_models, model, neighbors=25
)
token_timeline_low_dim_df.query("label=='main'")

global_distance, local_distance = plot_local_global_distances(
    token_timeline_df, token="p53"
)

global_distance

local_distance

g = plot_token_timeline(token_timeline_low_dim_df)
g.save(f"{str(timeline_folder)}/p53_time_plot.png")
print(g)

plot_wordcloud_neighbors_gif(
    token_timeline_low_dim_df, plot_filename=f"{str(wordcloud_folder)}/p53.gif"
)

# ![p53 gif here](output/wordcloud_plots_2000_2020/p53.gif)

plot_wordcloud_neighbors(
    token_timeline_low_dim_df, plot_filename=f"{str(wordcloud_folder)}/p53.png"
)

# ## Cystic

token_timeline_df = generate_timeline(year_comparison_dict, "cystic")
token_timeline_df.head()

token_timeline_low_dim_df = project_token_timeline(
    "cystic", aligned_models, model, neighbors=25
)
token_timeline_low_dim_df.query("label=='main'")

global_distance, local_distance = plot_local_global_distances(
    token_timeline_df, token="cystic"
)

global_distance

local_distance

g = plot_token_timeline(token_timeline_low_dim_df)
g.save(f"{str(timeline_folder)}/cystic_time_plot.png")
print(g)

plot_wordcloud_neighbors_gif(
    token_timeline_low_dim_df,
    plot_filename=f"{str(wordcloud_folder)}/cystic.gif",
)

# ![cystic_gif here](output/wordcloud_plots_2000_2020/cystic.gif)

plot_wordcloud_neighbors(
    token_timeline_low_dim_df,
    plot_filename=f"{str(wordcloud_folder)}/cystic.png",
)

# ## Cell

token_timeline_df = generate_timeline(year_comparison_dict, "cell")
token_timeline_df.head()

token_timeline_low_dim_df = project_token_timeline(
    "cell", aligned_models, model, neighbors=25
)
token_timeline_low_dim_df.query("label=='main'")

global_distance, local_distance = plot_local_global_distances(
    token_timeline_df, token="cell"
)

global_distance

local_distance

g = plot_token_timeline(token_timeline_low_dim_df)
g.save(f"{str(timeline_folder)}/celltime_plot.png")
print(g)

plot_wordcloud_neighbors_gif(
    token_timeline_low_dim_df, plot_filename=f"{str(wordcloud_folder)}/cell.gif"
)

# ![cell gif here](output/wordcloud_plots_2000_2020/cell.gif)

plot_wordcloud_neighbors(
    token_timeline_low_dim_df, plot_filename=f"{str(wordcloud_folder)}/cell.png"
)

# # Conclusions - Take Home Point(s)

# 1. Aligning vectors helps tremendously when measuring how words change their semantic meaning. Interesting spikes in the distance is words such as cystic, interleukin-18 and like 95%ci.
# 2. Plotting word clouds for the neighbor of each word highlights the shift these vectors are capturing
# 3. SpaCy likes to break hyphened words apart which makes capturing words such as RNA-seq and single-cell impossible to detect. Will need to update that if I want to have those words incorporated. Plus I need to use named entity recognition (NER tagger) to group nouns together as a high portion of biological terms are two words and not one.
