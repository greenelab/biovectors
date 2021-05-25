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

# # Generate UMAP GIF for Token Evolution Progression

# +
# %load_ext autoreload
# %autoreload 2

import itertools
from pathlib import Path
import pickle

from adjustText import adjust_text
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import tqdm

from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
# -

# # Load the data for Observing Change

decade_folder = Path("../word2vec_decade_experiment/output")
aligned_models = pickle.load(
    open(f"{str(decade_folder)}/aligned_word_vectors_2000_2020.pkl", "rb")
)

year_comparison_dict = {
    "_".join(comparison_file.stem.split("_")[0:2]): (
        pd.read_csv(str(comparison_file), sep="\t")
    )
    for comparison_file in (
        list(Path(f"{str(decade_folder)}/year_distances_2005_2020").rglob("*tsv"))
    )
}
list(year_comparison_dict.keys())[0:3]

year_comparison_dict["2005_2020"].sort_values("global_dist", ascending=False)

# # Train the UMAP Model

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

# ## Neighbors vs Min Distance

n_neighbors = [15, 25]
min_dist = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
years_to_test = [2000, 2005, 2010, 2015, 2020]

for neighbor, min_d in itertools.product(n_neighbors, min_dist):

    if Path(f"output/param_sweep_results/{neighbor}_{min_d}_results.tsv").exists():
        continue

    model = ParametricUMAP(
        verbose=False,
        metric="cosine",
        random_state=100,
        low_memory=True,
        n_neighbors=neighbor,
        min_dist=min_d,
    )
    embedding = model.fit_transform(word_models_stacked)
    embedding_model_df = pd.DataFrame(
        {
            "umap1": embedding[:, 0],
            "umap2": embedding[:, 1],
            "year": itertools.chain(
                *list(
                    map(
                        lambda x: aligned_models[x]
                        .query(f"token in {origin_df.token.tolist()}")
                        .sort_values("token")
                        .set_index("token")
                        .assign(year=x)
                        .year.tolist(),
                        aligned_models.keys(),
                    )
                )
            ),
            "token": itertools.chain(
                *list(
                    map(
                        lambda x: x.query(f"token in {origin_df.token.tolist()}")
                        .sort_values("token")
                        .set_index("token")
                        .index.tolist(),
                        aligned_models.values(),
                    )
                )
            ),
        }
    )

    embedding_model_df.to_csv(
        f"output/param_sweep_results/{neighbor}_{min_d}_results.tsv",
        sep="\t",
        index=False,
    )

grid_results = dict()
for file in Path("output/param_sweep_results").rglob("*tsv"):
    grid_results[tuple(file.stem.split("_")[0:2])] = pd.read_csv(str(file), sep="\t")

plot_df = pd.DataFrame(
    [], columns=["umap1", "umap2", "year", "token", "neighbor", "min_d"]
)
for key in grid_results:
    plot_df = plot_df.append(
        grid_results[key]
        .query(f"year in {years_to_test}")
        .groupby("year")
        .apply(lambda x: x.sample(1000, random_state=100))
        .reset_index(drop=True)
        .assign(neighbor=key[0], min_d=key[1])
    )

g = (
    p9.ggplot(plot_df, p9.aes(x="umap1", y="umap2", color="year"))
    + p9.geom_point()
    + p9.facet_grid("neighbor ~ min_d", labeller="label_both", scales="free")
    + p9.theme(figure_size=(12, 12))
    + p9.scale_color_brewer(type="qual", palette=3)
)
print(g)

# ## Epochs Test

n_neighbors = [25]
n_epochs = [2, 3, 5]
min_dist = [0.0]

for neighbor, min_d, epoch in itertools.product(n_neighbors, min_dist, n_epochs):

    if Path(
        f"output/param_sweep_results/{neighbor}_{min_d}_{epoch}_results.tsv"
    ).exists():
        continue

    model = ParametricUMAP(
        verbose=False,
        metric="cosine",
        random_state=100,
        low_memory=True,
        n_neighbors=neighbor,
        min_dist=min_d,
        n_training_epochs=epoch,
    )
    embedding = model.fit_transform(word_models_stacked)
    embedding_model_df = pd.DataFrame(
        {
            "umap1": embedding[:, 0],
            "umap2": embedding[:, 1],
            "year": itertools.chain(
                *list(
                    map(
                        lambda x: aligned_models[x]
                        .query(f"token in {origin_df.token.tolist()}")
                        .sort_values("token")
                        .set_index("token")
                        .assign(year=x)
                        .year.tolist(),
                        aligned_models.keys(),
                    )
                )
            ),
            "token": itertools.chain(
                *list(
                    map(
                        lambda x: x.query(f"token in {origin_df.token.tolist()}")
                        .sort_values("token")
                        .set_index("token")
                        .index.tolist(),
                        aligned_models.values(),
                    )
                )
            ),
        }
    )

    embedding_model_df.to_csv(
        f"output/param_sweep_results/{neighbor}_{min_d}_{epoch}_results.tsv",
        sep="\t",
        index=False,
    )

grid_results = dict()
for file in Path("output/param_sweep_results").rglob("*tsv"):
    params = file.stem.split("_")[:-1]

    if len(params) < 3:
        continue

    grid_results[tuple(file.stem.split("_")[0:3])] = pd.read_csv(str(file), sep="\t")

plot_df = pd.DataFrame(
    [], columns=["umap1", "umap2", "year", "token", "neighbor", "min_d"]
)
for key in grid_results:
    plot_df = plot_df.append(
        grid_results[key]
        .query(f"year in {years_to_test}")
        .groupby("year")
        .apply(lambda x: x.sample(1000, random_state=100))
        .reset_index(drop=True)
        .assign(neighbor=key[0], min_d=key[1], epoch=key[2])
    )

g = (
    p9.ggplot(plot_df, p9.aes(x="umap1", y="umap2", color="year"))
    + p9.geom_point()
    + p9.facet_grid("epoch ~ .", labeller="label_both", scales="free")
    + p9.theme(figure_size=(12, 12))
    + p9.scale_color_brewer(type="qual", palette=3)
)
print(g)
