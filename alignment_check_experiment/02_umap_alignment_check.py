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

# This notebook is designed to test a potential issue with the Word2Vec alignment algorithm. In the notebook [03_word_decade_figure_generator_2000_2020.ipynb](../word2vec_decade_experiment/03_word_decade_figure_generator_2000_2020.ipynb), words that shouldn't move a lot appear to be moving a lot within the early years (2000-2005). This led to the hypothesis that the alignment for word vectors doesn't work for the earlier years as there is a small amount of data. To see if there is an alignment issue I generated a umap plot of all tokens shared between 2000-2020 and color coded each point based on the year. Conclusion to be discussed at the bottom.

# +
# %load_ext autoreload
# %autoreload 2

import itertools
from pathlib import Path
import pickle

from adjustText import adjust_text
from gensim.models import Word2Vec
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import tqdm

from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP


# -

def plot_umap_years(embed_df, years):

    plot_df = (
        embed_df.query(f"year in {years}")
        .groupby("year")
        .apply(lambda x: x.sample(1000, random_state=100))
        .reset_index(drop=True)
    )

    g = (
        p9.ggplot(plot_df, p9.aes(x="umap1", y="umap2", color="year"))
        + p9.geom_point()
        + p9.theme(figure_size=(8, 6))
        + p9.scale_color_brewer(type="qual", palette=3)
    )
    return g


# # Load the data

year_cutoff = 2000
latest_year = 2020
aligned_models = dict()
decade_folder = Path("../word2vec_decade_experiment/output")

# Skip 2021 as that model is too small to analyze
# Try again December 2021
word_models = filter(
    lambda x: int(x.stem.split("_")[1]) >= year_cutoff
    and int(x.stem.split("_")[1]) != 2021,
    list((decade_folder / Path("models")).rglob("*model")),
)
word_models = sorted(word_models, key=lambda x: int(x.stem.split("_")[1]), reverse=True)
print(word_models)

for model in tqdm.tqdm(word_models):
    year = model.stem.split("_")[1]
    word_model = Word2Vec.load(str(model))

    if year == str(latest_year):
        origin_tokens = sorted(list(set(word_model.wv.vocab.keys())))
        remaining_tokens = origin_tokens

    else:
        tokens = sorted(list(set(word_model.wv.vocab.keys())))
        remaining_tokens = set(origin_tokens) & set(tokens)

    data_records = []

    for tok in remaining_tokens:
        data_entry = dict(
            zip(
                [f"feat_{idx}" for idx in range(len(word_model.wv[tok]))],
                word_model.wv[tok],
            )
        )
        data_entry["token"] = tok
        data_records.append(data_entry)

    aligned_models[year] = pd.DataFrame.from_records(data_records)

# # Train the UMAP Model without Alignment

# ## Train Model on Un-Aligned Matricies

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
file_name = "output/2000_2020_umap_model_no_alignment"

if not Path(file_name).exists():
    model = ParametricUMAP(
        verbose=False,
        metric="cosine",
        random_state=100,
        low_memory=True,
        n_neighbors=25,
        min_dist=0.0,
    )
    embedding = model.fit_transform(word_models_stacked)
    model.save(file_name)
else:
    model = load_ParametricUMAP(str(file_name))
    model.verbose = False
    embedding = model.transform(word_models_stacked)

unaligned_embedding_model_df = pd.DataFrame(
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
unaligned_embedding_model_df.sample(10, random_state=100)

# ## Plot Early Years vs Later Years

years_to_test = ["2000", "2003", "2005", "2010", "2015", "2020"]
plot_umap_years(unaligned_embedding_model_df, years_to_test)

# ## Plot Across the Years

years_to_test = ["2000", "2001", "2002", "2003", "2004", "2005"]
plot_umap_years(unaligned_embedding_model_df, years_to_test)

years_to_test = ["2006", "2007", "2008", "2009", "2010", "2011"]
plot_umap_years(unaligned_embedding_model_df, years_to_test)

years_to_test = ["2012", "2013", "2014", "2015", "2016", "2017"]
plot_umap_years(unaligned_embedding_model_df, years_to_test)

years_to_test = ["2018", "2019", "2020"]
plot_umap_years(unaligned_embedding_model_df, years_to_test)

# # Train UMAP with Alignment

# ## Train Model on Aligned Matricies

aligned_models = pickle.load(
    open(str(decade_folder / Path("aligned_word_vectors_2000_2020.pkl")), "rb")
)

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
file_name = "output/2000_2020_umap_model_with_alignment"

if not Path(file_name).exists():
    model = ParametricUMAP(
        verbose=False,
        metric="cosine",
        random_state=100,
        low_memory=True,
        n_neighbors=25,
        min_dist=0.0,
    )
    embedding = model.fit_transform(word_models_stacked)
    model.save(file_name)
else:
    model = load_ParametricUMAP(str(file_name))
    model.verbose = False
    embedding = model.transform(word_models_stacked)

aligned_embedding_model_df = pd.DataFrame(
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
aligned_embedding_model_df.sample(10, random_state=100)

# ## Plot the early years against the later years

years_to_test = ["2000", "2003", "2005", "2010", "2015", "2020"]
plot_umap_years(aligned_embedding_model_df, years_to_test)

# ## Plot each Year Along the Axes

years_to_test = ["2000", "2001", "2002", "2003", "2004", "2005"]
plot_umap_years(aligned_embedding_model_df, years_to_test)

years_to_test = ["2006", "2007", "2008", "2009", "2010", "2011"]
plot_umap_years(aligned_embedding_model_df, years_to_test)

years_to_test = ["2012", "2013", "2014", "2015", "2016", "2017"]
plot_umap_years(aligned_embedding_model_df, years_to_test)

years_to_test = ["2018", "2019", "2020", "2021"]
plot_umap_years(aligned_embedding_model_df, years_to_test)

# # 'Are' use case for aligned and unaligned

plot_df = (
    aligned_embedding_model_df.query("token=='are'")
    .assign(label="align")
    .append(unaligned_embedding_model_df.query("token=='are'").assign(label="unalign"))
)
plot_df.sample(10, random_state=100)

# ## Plot both versions

g = (
    p9.ggplot(plot_df, p9.aes(x="umap1", y="umap2", color="label", label="year"))
    + p9.geom_text()
    + p9.scale_color_brewer(type="qual", palette=2)
    + p9.facet_grid("label ~ .", scales="free")
)
print(g)

# # 'interleukin-18' use case for aligned and unaligned

plot_df = (
    aligned_embedding_model_df.query("token=='interleukin-18'")
    .assign(label="align")
    .append(
        unaligned_embedding_model_df.query("token=='interleukin-18'").assign(
            label="unalign"
        )
    )
)
plot_df.sample(10, random_state=100)

# ## Plot both versions

g = (
    p9.ggplot(plot_df, p9.aes(x="umap1", y="umap2", color="label", label="year"))
    + p9.geom_text()
    + p9.scale_color_brewer(type="qual", palette=2)
    + p9.facet_grid("label ~ .", scales="free")
)
print(g)

# # Conclusions and Take Home Points

# 1. The alignment process seems to be working as the scale is a lot more reasonable once aligned to 2020.
# 2. The shift scale for words is more reasonable after alignment, which provides indication that alignment does work in the grand scheme of this project; however, pre-2005 may still be an issue.
# 3. The word are actually doesn't move a lot when compared to the unaligned version, which is a great negative control for this experiment.
# 4. The gene interleukin-18 underwent a larger change than 'are' which makes sense as this gene was discovered to be an indicator for inflammation for certain diseases.
