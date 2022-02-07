# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # Parameter Sweep for UMAP

# The last piece of this puzzle is to generate a visualization to show how a word is shifting through time.
# In order to capture this sort of information the choice of model is ParametricUMAP which is a hybrid of the regular UMAP algorithm and neural networks.
# The rationale for this mix is that a neural network is trained to model the neighborhood graph that UMAP needs to generate.
# The best part about using a network is that one doesn't have to deal with memory issues that will arise within this data (300 dim * 100k+ words * 10 years)
# Before we use ParametricUMAP, it's important to work with the original umap to see how words cluster together.

# +
import itertools
from pathlib import Path
import re

from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pandas as pd
import plotnine as p9
import plydata as ply
import plydata.tidy as ply_tdy
import tqdm
import umap
# -

# # Unaligned Models

# ## Load the paths for each year

# The goal here is to grab the odd years unaligned and see if umap can tease apart odd years.

unaligned_word_models = list(
    Path("../multi_model_experiment/output/models").rglob("*model")
)
unaligned_word_models = sorted(unaligned_word_models, key=lambda x: x.stem)
unaligned_word_model_filter = list(
    filter(lambda x: "_0" in x.stem, unaligned_word_models)
)

word_freq_count_cutoff = 5
odd_year_subset = unaligned_word_model_filter[11:-1:2]

training_unaligned_word_model_map = dict()
for word_file in tqdm.tqdm(odd_year_subset):
    model = Word2Vec.load(str(word_file)).wv
    training_unaligned_word_model_map[word_file.stem.split("_")[0]] = dict(
        model=model,
        cutoff_index=min(
            map(
                lambda x: 999999
                if model.get_vecattr(x[1], "count") > word_freq_count_cutoff
                else x[0],
                enumerate(model.index_to_key),
            )
        ),
    )

words_to_visualize = []
token_character_list = []
year_labels_list = []

for year in tqdm.tqdm(training_unaligned_word_model_map):
    model = training_unaligned_word_model_map[year]["model"]
    word_subset_matrix = model[
        model.index_to_key[: training_unaligned_word_model_map[year]["cutoff_index"]]
    ]
    print((year, word_subset_matrix.shape))
    words_to_visualize.append(word_subset_matrix)
    token_character_list += list(
        map(
            lambda x: re.escape(x),
            model.index_to_key[
                : training_unaligned_word_model_map[year]["cutoff_index"]
            ],
        )
    )
    year_labels_list += [year] * len(
        model.index_to_key[: training_unaligned_word_model_map[year]["cutoff_index"]]
    )

training_unaligned_words = np.vstack(words_to_visualize)
training_unaligned_words

# ## Run the Parameter Sweep

n_neighbors = [10, 25, 50, 75]
epoch_multiplier = [1]
param_output = []

for neighbors, epoch_m in itertools.product(n_neighbors, epoch_multiplier):
    file_name = f"output/umap_parameter_sweep/unaligned/neighbors({neighbors})_epochs({epoch_m})"

    if Path(f"{file_name}.tsv").exists():
        continue

    print(neighbors, epoch_m)

    model = umap.UMAP(
        verbose=True,
        metric="cosine",
        random_state=101,
        low_memory=True,
        n_neighbors=neighbors,
        min_dist=0.99,
        n_epochs=epoch_m * 10,
    )

    umap_embeddings = model.fit_transform(training_unaligned_words)

    (
        pd.DataFrame(umap_embeddings, columns=["umap1", "umap2"])
        >> ply.define(year=year_labels_list, tok=token_character_list)
        >> ply.define(neighbors=neighbors, epochs=epoch_m)
        >> ply.call(
            ".to_csv",
            f"output/umap_parameter_sweep/unaligned/neighbors({neighbors})_epochs({epoch_m}).tsv",
            sep="\t",
            index=False,
        )
    )

# ## Visualize Embeddings

# Now that we have generated each embedding its time to visually confirm the years can be separated.

plot_df = pd.concat(
    [
        pd.read_csv(str(file), sep="\t")
        >> ply.call(".dropna")
        >> ply.define(umap1="umap1.astype(float)")
        for file in Path("output/umap_parameter_sweep/unaligned").rglob("*tsv")
    ]
)
plot_df >> ply.slice_rows(20)

g = (
    p9.ggplot(
        plot_df >> ply.define(year="pd.Categorical(year)"),
        p9.aes(x="umap1", y="umap2", fill="year"),
    )
    + p9.geom_point(alpha=0.2)
    + p9.labs(title="Parameter Sweep of UMAP 2011-2020 (odd years)")
    + p9.theme(figure_size=(10, 8))
    + p9.facet_wrap("neighbors")
)
print(g)

# # Aligned Models

# ## Load the paths for each year

# The goal here is to grab the odd years unaligned and see if umap can tease apart odd years.

aligned_word_models = list(
    Path("../multi_model_experiment/output/aligned_vectors_tmp").rglob("*kv")
)
aligned_word_models = sorted(aligned_word_models, key=lambda x: x.stem)
aligned_word_model_filter = list(filter(lambda x: "_0" in x.stem, aligned_word_models))

word_freq_count_cutoff = 1
odd_year_subset = aligned_word_model_filter[11:-1:2]

aligned_training_word_model_map = dict()
for word_file in tqdm.tqdm(odd_year_subset):
    model = KeyedVectors.load(str(word_file))
    aligned_training_word_model_map[word_file.stem.split("_")[0]] = dict(
        model=model,
        cutoff_index=min(
            map(
                lambda x: 999999
                if model.get_vecattr(x[1], "count") > word_freq_count_cutoff
                else x[0],
                enumerate(model.index_to_key),
            )
        ),
    )

words_to_visualize = []
token_character_list = []
year_labels_list = []

for year in tqdm.tqdm(aligned_training_word_model_map):
    model = aligned_training_word_model_map[year]["model"]
    word_subset_matrix = model[model.index_to_key]
    print((year, word_subset_matrix.shape))
    words_to_visualize.append(word_subset_matrix)
    token_character_list += list(map(lambda x: re.escape(x), model.index_to_key))
    year_labels_list += [year] * len(model.index_to_key)

aligned_training_words = np.vstack(words_to_visualize)
aligned_training_words.shape

# ## Run the Parameter Sweep

n_neighbors = [50]
epoch_multiplier = [1]
param_output = []

for neighbors, epoch_m in itertools.product(n_neighbors, epoch_multiplier):
    file_name = (
        f"output/umap_parameter_sweep/aligned/neighbors({neighbors})_epochs({epoch_m})"
    )

    if Path(f"{file_name}.tsv").exists():
        continue

    print(neighbors, epoch_m)

    model = umap.UMAP(
        verbose=True,
        metric="cosine",
        random_state=101,
        low_memory=True,
        n_neighbors=neighbors,
        min_dist=0.99,
        n_epochs=epoch_m * 10,
    )

    umap_embeddings = model.fit_transform(aligned_training_words)

    (
        pd.DataFrame(umap_embeddings, columns=["umap1", "umap2"])
        >> ply.define(year=year_labels_list, tok=token_character_list)
        >> ply.define(neighbors=neighbors, epochs=epoch_m)
        >> ply.call(
            ".to_csv",
            f"output/umap_parameter_sweep/aligned/neighbors({neighbors})_epochs({epoch_m}).tsv",
            sep="\t",
            index=False,
        )
    )

# ## Visualize Embeddings

# Now that we have generated each embedding its time to visually confirm the years can be separated.

plot_df = pd.concat(
    [
        pd.read_csv(str(file), sep="\t")
        >> ply.call(".dropna")
        >> ply.define(umap1="umap1.astype(float)")
        for file in Path("output/umap_parameter_sweep/aligned").rglob("*tsv")
    ]
)
plot_df >> ply.slice_rows(20)

g = (
    p9.ggplot(
        plot_df >> ply.define(year="pd.Categorical(year)"),
        p9.aes(x="umap1", y="umap2", fill="year"),
    )
    + p9.geom_point(alpha=0.2)
    + p9.labs(title="Parameter Sweep of UMAP 2011-2020 (odd years)")
    + p9.theme(figure_size=(10, 8))
    + p9.facet_wrap("neighbors")
)
print(g)

# # Take Home Messages

# 1. UMAP is able to capture different clusters across the odd years.
# 2. The more neighbors one uses for the umap algorithm the more distinct the clusters for unaligned word vector models.
# 3. After alignment it does appear that orthogonal procustes is able to fix this issue by allowing words from the different years be placed closer together.
# 4. Moving forward onto the next experiment, I'll be using neighbor values of at least 20.
