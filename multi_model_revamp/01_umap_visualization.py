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

# # UMAP Visualization for Word2Vec

# +
from pathlib import Path

from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pandas as pd
import plotnine as p9
import plydata as ply
import tensorflow as tf
import umap
# -

# ## Preselect tokens for Projections

selected_year = "2010"
tokens_sampled = 5000

unaligned_models = list(Path(f"output/models/{selected_year}_model").glob("*model"))
aligned_models = list(Path(f"output/aligned_models/{selected_year}_model").glob("*kv"))
output_file_folder = Path("output/figure_data_and_figures/alignment_visualization")

np.random.seed(100)
total_matching_tokens = set()
for model_path in aligned_models:
    word_model_to_query = KeyedVectors.load(str(model_path))
    if len(total_matching_tokens) == 0:
        total_matching_tokens = set(word_model_to_query.vocab.keys())
    else:
        total_matching_tokens &= set(word_model_to_query.vocab.keys())
total_matching_tokens = list(total_matching_tokens)

total_words = len(total_matching_tokens)
token_indicies = np.random.choice(total_words, tokens_sampled)
selected_tokens = [total_matching_tokens[idx] for idx in token_indicies]

np.random.seed(100)
example_tok = np.random.choice(sorted(selected_tokens), size=1)[0]
example_tok

# ## Unaligned

# +
word_model_list = list()
for model_path in unaligned_models:
    year = model_path.stem.split("_")[0]
    word_model_list.append(model_path)

word_model_list = sorted(word_model_list, key=lambda x: int(x.stem.split("_")[1]))
word_model_list
# -

word_vectors = []
unaligned_tokens = []
for idx, model_path in enumerate(word_model_list):
    word2vec_model = Word2Vec.load(str(model_path))
    word_vectors.append(word2vec_model.wv[selected_tokens])
    unaligned_tokens += selected_tokens

    # only use five models
    if idx > 4:
        break

tf.random.set_seed(100)
umap_model = umap.UMAP(
    verbose=True,
    metric="cosine",
    random_state=100,
    low_memory=True,
    n_neighbors=25,
    min_dist=0.99,
    n_epochs=50,
)
unaligned_model_embeddings = umap_model.fit_transform(np.vstack(word_vectors))

unaligned_tok_df = (
    pd.DataFrame(unaligned_model_embeddings, columns=["umap1", "umap2"])
    >> ply.define(tok=unaligned_tokens)
    >> ply.define(
        year_label=(
            ["0"] * tokens_sampled
            + ["1"] * tokens_sampled
            + ["2"] * tokens_sampled
            + ["3"] * tokens_sampled
            + ["4"] * tokens_sampled
            + ["5"] * tokens_sampled
        )
    )
)
unaligned_tok_df >> ply.call(
    ".to_csv",
    f"{str(output_file_folder)}/unaligned_{selected_year}_umap_tokens.tsv",
    sep="\t",
    index=False,
)
unaligned_tok_df >> ply.slice_rows(10)

g = unaligned_tok_df >> (
    p9.ggplot()
    + p9.aes(x="umap1", y="umap2", fill="year_label")
    + p9.geom_point(alpha=0.2)
    + p9.scale_fill_brewer(type="qual", palette="Dark2")
    + p9.labs(
        title=f"{tokens_sampled:,} randomly sampled tokens in {selected_year} pre-alignment",
        fill="Model Index",
    )
)
print(g)
g.save(f"{str(output_file_folder)}/unaligned_{selected_year}_plot.svg")
g.save(f"{str(output_file_folder)}/unaligned_{selected_year}_plot.png", dpi=300)

# +
g = (
    unaligned_tok_df
    >> ply.query(f"tok=='{example_tok}'")
    >> (
        p9.ggplot()
        + p9.aes(x="umap1", y="umap2", fill="year_label")
        + p9.geom_point()
        + p9.scale_fill_brewer(type="qual", palette="Dark2")
        + p9.geom_point(
            p9.aes(x="umap1", y="umap2"), data=unaligned_tok_df, fill="grey", alpha=0.01
        )
        + p9.labs(
            title=f"'{example_tok}' in {selected_year} pre-alignemnt",
            fill="Model Index",
        )
    )
)

print(g)
g.save(f"{str(output_file_folder)}/unaligned_{selected_year}_{example_tok}_plot.svg")
g.save(
    f"{str(output_file_folder)}/unaligned_{selected_year}_{example_tok}_plot.svg",
    dpi=300,
)
# -

# ## Aligned

aligned_word_vectors = []
aligned_tokens = []
for idx, model_path in enumerate(aligned_models):
    word2vec_model = KeyedVectors.load(str(model_path))
    aligned_word_vectors.append(word2vec_model.wv[selected_tokens])
    aligned_tokens += selected_tokens

    # only use four models
    if idx > 4:
        break

umap_model = umap.UMAP(
    verbose=True,
    metric="cosine",
    random_state=101,
    low_memory=True,
    n_neighbors=25,
    min_dist=0.99,
    n_epochs=50,
)
aligned_model_embeddings = umap_model.fit_transform(np.vstack(aligned_word_vectors))

aligned_tok_df = (
    pd.DataFrame(aligned_model_embeddings, columns=["umap1", "umap2"])
    >> ply.define(tok=aligned_tokens)
    >> ply.define(
        year_label=(
            ["0"] * tokens_sampled
            + ["1"] * tokens_sampled
            + ["2"] * tokens_sampled
            + ["3"] * tokens_sampled
            + ["4"] * tokens_sampled
            + ["5"] * tokens_sampled
        )
    )
)
aligned_tok_df >> ply.call(
    ".to_csv",
    f"{str(output_file_folder)}/aligned_{selected_year}_umap_tokens.tsv",
    sep="\t",
    index=False,
)
aligned_tok_df >> ply.slice_rows(10)

g = aligned_tok_df >> (
    p9.ggplot()
    + p9.aes(x="umap1", y="umap2", fill="year_label")
    + p9.geom_point(alpha=0.2)
    + p9.scale_fill_brewer(type="qual", palette="Dark2")
    + p9.labs(
        title=f"{tokens_sampled:,} randomly sampled tokens in {selected_year} post-alignment",
        fill="Model Index",
    )
)
print(g)
g.save(f"{str(output_file_folder)}/aligned_{selected_year}_plot.svg")
g.save(f"{str(output_file_folder)}/aligned_{selected_year}_plot.png", dpi=300)

g = (
    aligned_tok_df
    >> ply.query(f"tok=='{example_tok}'")
    >> (
        p9.ggplot()
        + p9.aes(x="umap1", y="umap2", fill="year_label")
        + p9.geom_point()
        + p9.scale_fill_brewer(type="qual", palette="Dark2")
        + p9.geom_point(
            p9.aes(x="umap1", y="umap2"), data=aligned_tok_df, fill="grey", alpha=0.01
        )
        + p9.labs(
            title=f"'{example_tok}' in {selected_year} post-alignemnt",
            fill="Model Index",
        )
    )
)
print(g)
g.save(f"{str(output_file_folder)}/aligned_{selected_year}_{example_tok}_plot.svg")
g.save(
    f"{str(output_file_folder)}/aligned_{selected_year}_{example_tok}_plot.png", dpi=300
)
