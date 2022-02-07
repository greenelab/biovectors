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

# # Parametric UMAP vs Align UMAP

# +
from collections import OrderedDict
import itertools
from pathlib import Path
import pickle
import re

from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pandas as pd
import plotnine as p9
import plydata as ply
import plydata.tidy as ply_tdy
import tqdm

from umap.aligned_umap import AlignedUMAP
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP

from biovectors_modules.word2vec_analysis_helper import window
# -

# # Parametric UMAP

# ## Unaligned Models

unaligned_word_models = list(
    Path("../multi_model_experiment/output/models").rglob("*model")
)
unaligned_word_models = sorted(unaligned_word_models, key=lambda x: x.stem)
unaligned_word_model_filter = list(
    filter(lambda x: "_0" in x.stem, unaligned_word_models)
)

word_freq_count_cutoff = 5
odd_year_subset = unaligned_word_model_filter[15:-1:2]

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
training_unaligned_words.shape

umap_model_path = Path("output/unaligned_parametric_umap_model")
if not umap_model_path.exists():
    unaligned_parametric_model = ParametricUMAP(
        verbose=True,
        metric="cosine",
        random_state=101,
        low_memory=True,
        n_neighbors=25,
        min_dist=0.99,
        n_training_epochs=1,
    )
    umap_embeddings = unaligned_parametric_model.fit_transform(training_unaligned_words)
    unaligned_parametric_model.save(str(umap_model_path))
else:
    unaligned_parametric_model = load_ParametricUMAP(umap_model_path)
    umap_embeddings = unaligned_parametric_model.transform(training_unaligned_words)

train_plot_df = (
    pd.DataFrame(umap_embeddings, columns=["umap1", "umap2"])
    >> ply.define(year=year_labels_list, tok=token_character_list)
    >> ply.call(".dropna")
    >> ply.define(umap1="umap1.astype(float)")
)
train_plot_df >> ply.slice_rows(10)

g = (
    p9.ggplot(
        train_plot_df >> ply.define(year="pd.Categorical(year)"),
        p9.aes(x="umap1", y="umap2", fill="year"),
    )
    + p9.geom_point(alpha=0.2)
    + p9.labs(title="Parameter Sweep of UMAP 2011-2020 (odd years)")
    + p9.theme(figure_size=(10, 8))
)
print(g)

# ## Project held out data onto the training years

validation_word_model_map = dict()
for word_file in tqdm.tqdm(unaligned_word_model_filter[10:15]):
    model = Word2Vec.load(str(word_file)).wv
    validation_word_model_map[word_file.stem.split("_")[0]] = dict(
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

for year in tqdm.tqdm(validation_word_model_map):
    model = validation_word_model_map[year]["model"]
    word_subset_matrix = model[
        model.index_to_key[: validation_word_model_map[year]["cutoff_index"]]
    ]
    print((year, word_subset_matrix.shape))
    words_to_visualize.append(word_subset_matrix)
    token_character_list += list(
        map(
            lambda x: re.escape(x),
            model.index_to_key[: validation_word_model_map[year]["cutoff_index"]],
        )
    )
    year_labels_list += [year] * len(
        model.index_to_key[: validation_word_model_map[year]["cutoff_index"]]
    )

umap_embeddings = unaligned_parametric_model.transform(np.vstack(words_to_visualize))

val_plot_df = (
    pd.DataFrame(umap_embeddings, columns=["umap1", "umap2"])
    >> ply.define(year=year_labels_list, tok=token_character_list)
    >> ply.call(".dropna")
    >> ply.define(umap1="umap1.astype(float)")
)
val_plot_df >> ply.slice_rows(10)

g = (
    p9.ggplot(
        val_plot_df >> ply.define(year="pd.Categorical(year)"),
        p9.aes(x="umap1", y="umap2", fill="year"),
    )
    + p9.geom_point(alpha=0.1)
    + p9.labs(title="Parameter Sweep of UMAP 2011-2020 (odd years)")
    + p9.theme(figure_size=(10, 8))
    + p9.facet_wrap("year")
)
print(g)

# ## Aligned Models

aligned_word_models = list(
    Path("../multi_model_experiment/output/aligned_vectors_tmp").rglob("*kv")
)
aligned_word_models = sorted(aligned_word_models, key=lambda x: x.stem)
aligned_word_model_filter = list(filter(lambda x: "_0" in x.stem, aligned_word_models))

word_freq_count_cutoff = 1
odd_year_subset = aligned_word_model_filter[17:-1:2]

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

umap_model_path = Path("output/aligned_parametric_umap_model")
if not umap_model_path.exists():
    aligned_parametric_model = ParametricUMAP(
        verbose=True,
        metric="cosine",
        random_state=101,
        low_memory=True,
        n_neighbors=20,
        min_dist=0.99,
        n_training_epochs=1,
    )
    umap_embeddings = aligned_parametric_model.fit_transform(aligned_training_words)
    aligned_parametric_model.save(str(umap_model_path))
else:
    aligned_parametric_model = load_ParametricUMAP(umap_model_path)
    umap_embeddings = aligned_parametric_model.transform(aligned_training_words)

aligned_train_plot_df = (
    pd.DataFrame(umap_embeddings, columns=["umap1", "umap2"])
    >> ply.define(year=year_labels_list, tok=token_character_list)
    >> ply.call(".dropna")
    >> ply.define(umap1="umap1.astype(float)")
)
aligned_train_plot_df >> ply.slice_rows(10)

g = (
    p9.ggplot(
        aligned_train_plot_df >> ply.define(year="pd.Categorical(year)"),
        p9.aes(x="umap1", y="umap2", fill="year"),
    )
    + p9.geom_point(alpha=0.1)
    + p9.labs(title="Parametric UMAP 2011-2020")
    + p9.theme(figure_size=(10, 8))
)
print(g)

g = (
    p9.ggplot(
        aligned_train_plot_df >> ply.define(year="pd.Categorical(year)"),
        p9.aes(x="umap1", y="umap2", fill="year"),
    )
    + p9.geom_point(alpha=0.1)
    + p9.labs(title="Parameter Sweep of UMAP 2011-2020 (odd years)")
    + p9.theme(figure_size=(10, 8))
    + p9.facet_wrap("year")
)
print(g)

# # Aligned UMAP

word_freq_count_cutoff = 5
training_word_model_map = OrderedDict()

for word_file in tqdm.tqdm(unaligned_word_model_filter[10:]):
    model = Word2Vec.load(str(word_file)).wv
    year = word_file.stem.split("_")[0]

    training_word_model_map[year] = dict(
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

    training_word_model_map[year]["index_to_token_map"] = dict(
        zip(
            model.index_to_key[: training_word_model_map[year]["cutoff_index"]],
            range(
                len(model.index_to_key[: training_word_model_map[year]["cutoff_index"]])
            ),
        )
    )

relations = []
for year_one, year_two in tqdm.tqdm(window(training_word_model_map, 2)):
    year_one_map = training_word_model_map[year_one]["index_to_token_map"]
    year_two_map = training_word_model_map[year_two]["index_to_token_map"]
    relations.append(
        {
            year_one_map[tok]: year_two_map[tok]
            for tok in year_one_map
            if tok in year_two_map
        }
    )

words_to_visualize = []
token_character_list = []
year_labels_list = []

for year in tqdm.tqdm(training_word_model_map):
    model = training_word_model_map[year]["model"]
    word_subset_matrix = model[
        model.index_to_key[: training_word_model_map[year]["cutoff_index"]]
    ]
    print((year, word_subset_matrix.shape))
    words_to_visualize.append(word_subset_matrix)
    token_character_list += list(
        map(
            lambda x: re.escape(x),
            model.index_to_key[: training_word_model_map[year]["cutoff_index"]],
        )
    )
    year_labels_list += [year] * len(
        model.index_to_key[: training_word_model_map[year]["cutoff_index"]]
    )

aligned_umap_output_path = Path("output/aligned_umap_model.pkl")
if not aligned_umap_output_path.exists():
    # runtime - 3h 44m 17s
    aligned_umap_model = AlignedUMAP(
        n_components=2,
        verbose=True,
        metric="cosine",
        random_state=101,
        low_memory=True,
        n_neighbors=25,
        min_dist=0.99,
        alignment_window_size=2,
        alignment_regularisation=1e-3,
    )
    aligned_umap_model.fit(words_to_visualize, relations=relations)
    aligned_umap_model.embeddings_ = list(aligned_umap_model.embeddings_)
    pickle.dump(aligned_umap_model, open(str(aligned_umap_output_path), "wb"))
else:
    aligned_umap_model = pickle.load(open(str(aligned_umap_output_path), "rb"))

aligned_plot_df = pd.DataFrame(
    np.vstack(aligned_umap_model.embeddings_), columns=["umap1", "umap2"]
) >> ply.define(year="year_labels_list", tok="token_character_list")
aligned_plot_df >> ply.slice_rows(10)

g = (
    p9.ggplot(
        aligned_plot_df >> ply.define(year="pd.Categorical(year)"),
        p9.aes(x="umap1", y="umap2", fill="year"),
    )
    + p9.geom_point(alpha=0.05)
    + p9.labs(title="Aligned UMAP 2010-2020")
    + p9.theme(figure_size=(10, 8))
)
print(g)
