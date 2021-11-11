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

# # Change Point Detection

# This notebook is designed to perform bayesian change point detection on each individual token.
# This method provides probability estimates that a changepoint occurred at a given time point.
# There may be a problem with detecting changes at the extreme ends of the time series but inconclusive at the moment.

# +
from functools import partial
from pathlib import Path
import re

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import plotnine as p9
import plydata as ply
import plydata.tidy as ply_tdy
import tqdm

import bayesian_changepoint_detection.offline_changepoint_detection as offcd
# -

# # Grab Word Frequencies

if not Path("output/all_tok_frequencies.tsv").exists():
    data = []
    for model in tqdm.tqdm(
        sorted(
            list(Path("../multi_model_experiment/output/models").rglob("*/*_0.model"))
        )
    ):
        wv_model = Word2Vec.load(str(model))
        word_freq_df = pd.DataFrame.from_records(
            [
                {
                    "tok": tok,
                    "word_count": wv_model.wv.get_vecattr(tok, "count"),
                    "year": re.search(r"(\d+)_0", str(model)).group(1),
                }
                for tok in wv_model.wv.index_to_key
            ]
        )
        total_count = word_freq_df.word_count.sum()
        word_freq_df = word_freq_df >> ply.define(frequency=f"word_count/{total_count}")
        data.append(word_freq_df)

if not Path("output/all_tok_frequencies.tsv.xz").exists():
    all_word_freq_df = pd.concat(data)
    all_word_freq_df.dropna().to_csv(
        "output/all_tok_frequencies.tsv.xz", sep="\t", index=False, compression="xz"
    )
else:
    all_word_freq_df = (
        pd.read_csv("output/all_tok_frequencies.tsv.xz", sep="\t")
        .dropna()
        .astype({"year": int})
    )
all_word_freq_df

subsetted_tokens = pd.read_csv(
    "../multi_model_experiment/output/subsetted_tokens.tsv", sep="\t"
)
token_filter_list = subsetted_tokens.tok.tolist()
subsetted_tokens.head()

# ## Calculate percent change for frequency

all_word_pct_change_df = (
    all_word_freq_df
    >> ply.query(f"tok in {token_filter_list}")
    >> ply.group_by("tok")
    >> ply.arrange("year")
    >> ply.define(
        freq_pct_change="frequency.pct_change()",
        frequency_ratio=lambda x: x.frequency / x.shift(1).frequency,
    )
    >> ply.ungroup()
    >> ply.call(".dropna")
    >> ply.define(year=lambda x: x["year"].apply(lambda y: f"{int(y)-1}-{y}"))
)
all_word_pct_change_df.head()

# # Grab semantic change values

distance_files = list(
    Path("../multi_model_experiment/output/combined_inter_intra_distances").rglob(
        "saved_*_distance.tsv"
    )
)
print(len(distance_files))

year_distance_map = {
    re.search(r"\d+", str(year_file)).group(0): (pd.read_csv(str(year_file), sep="\t"))
    for year_file in tqdm.tqdm(distance_files)
}

full_token_set_df = pd.concat(
    [
        year_distance_map[year] >> ply.query("year_2-year_1 == 1")
        for year in tqdm.tqdm(year_distance_map)
    ]
)
print(full_token_set_df.shape)
full_token_set_df.head()

# # Combine both information

# Using an idea similar to SCAF \[[1](https://doi.org/10.1007/s00799-019-00271-6)\], I'm combining the global qst metric with the percent change in frequency.
# SCAF uses percent change for both metrics; however, the caveat is that their method loses information for the first two timepoints.
# In my case given that global_distance_qst is a metric that's bound between 0 and 1 frequency percent change can be used directly.
# By combining these two terms I can use this metric as a means to estimate change and allow for bayesian changepoint detection to calculate the probability of a timepoint change.

merged_frequency_df = (
    full_token_set_df
    >> ply_tdy.unite("year", "year_1", "year_2", sep="-")
    >> ply.inner_join(all_word_pct_change_df, on=["tok", "year"])
    >> ply.rename(year_pair="year")
    >> ply.select(
        "tok",
        "original_global_distance",
        "global_distance_qst",
        "ratio_metric",
        "year_pair",
        "frequency",
        "freq_pct_change",
        "frequency_ratio",
    )
)
merged_frequency_df

change_metric_df = (
    merged_frequency_df
    >> ply.group_by("tok")
    >> ply.arrange("year_pair")
    >> ply.define(
        change_metric_qst="global_distance_qst + freq_pct_change",
        change_metric_ratio="ratio_metric + frequency_ratio",
    )
    >> ply.ungroup()
)
change_metric_df

# # Change point detection

# Use Semantic Change Analysis with Frequency (SCAF) to perform bayesian change point detection.

if not Path("output/bayesian_changepoint_data.tsv", sep="\t").exists():
    change_point_results = []
    for tok, tok_series_df in tqdm.tqdm(change_metric_df.groupby("tok")):

        change_metric_ratio = np.insert(
            tok_series_df >> ply.pull("change_metric_ratio"), 0, 0
        )
        Q, P, Pcp = offcd.offline_changepoint_detection(
            change_metric_ratio,
            partial(offcd.const_prior, l=(len(change_metric_ratio) + 1)),
            offcd.gaussian_obs_log_likelihood,
            truncate=-40,
        )
        estimated_changepoint_probability = np.exp(Pcp).sum(0)

        change_point_results.append(
            pd.DataFrame.from_dict(
                {
                    "tok": [tok] * len(estimated_changepoint_probability),
                    "changepoint_prob": estimated_changepoint_probability,
                    "year_pair": (
                        tok_series_df
                        >> ply.select("year_pair")
                        >> ply.pull("year_pair")
                    ),
                }
            )
        )

if not Path("output/bayesian_changepoint_data.tsv", sep="\t").exists():
    change_point_df = pd.concat(change_point_results)
    change_point_df.to_csv(
        "output/bayesian_changepoint_data.tsv", sep="\t", index=False
    )
else:
    change_point_df = pd.read_csv("output/bayesian_changepoint_data.tsv", sep="\t")
change_point_df.head()

(change_point_df >> ply.arrange("-changepoint_prob") >> ply.slice_rows(30))

(
    p9.ggplot(change_point_df >> ply.query("tok=='coronavirus'"))
    + p9.aes(x="year_pair", y="changepoint_prob", group=0)
    + p9.geom_point()
    + p9.geom_line()
    + p9.coord_flip()
    + p9.theme_seaborn("white")
    + p9.labs(
        x="Year Shift",
        y="Probability of Changepoint",
        title="Changepoint Prediction for Token ('coronavirus')",
    )
)

(
    p9.ggplot(change_point_df >> ply.query("tok=='copyright'"))
    + p9.aes(x="year_pair", y="changepoint_prob", group=0)
    + p9.geom_point()
    + p9.geom_line()
    + p9.coord_flip()
    + p9.theme_seaborn("white")
    + p9.labs(
        x="Year Shift",
        y="Probability of Changepoint",
        title="Changepoint Prediction for Token ('copyright')",
    )
)

(
    p9.ggplot(change_point_df >> ply.query("tok=='pandemic'"))
    + p9.aes(x="year_pair", y="changepoint_prob", group=0)
    + p9.geom_point()
    + p9.geom_line()
    + p9.coord_flip()
    + p9.theme_seaborn("white")
    + p9.labs(
        x="Year Shift",
        y="Probability of Changepoint",
        title="Changepoint Prediction for Token ('pandemic')",
    )
)

(
    p9.ggplot(change_point_df >> ply.query("tok=='the'"))
    + p9.aes(x="year_pair", y="changepoint_prob", group=0)
    + p9.geom_point()
    + p9.geom_line()
    + p9.coord_flip()
    + p9.theme_seaborn("white")
    + p9.labs(
        x="Year Shift",
        y="Probability of Changepoint",
        title="Changepoint Prediction for Token ('the')",
    )
    + p9.scale_y_continuous(limits=[0, 1])
)

# # Take Home Points

# 1. Bayesian change point detection provides insight on the specific year period a semantic change point may have occurred.
# 2. Best positive result is pandemic which underwent a focus shift from bird flu and influenza to coronavirus.
# 3. Follow up analysis which will appear in the next notebook will involve looking at the top X token neighbors to the query word. By doing that one can estimate which kind of shift a word has undergone.
