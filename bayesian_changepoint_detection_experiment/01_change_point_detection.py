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

# This notebook is designed to perform bayesian change point detection and run the CUSUM algorithm on each individual token.
# The intended idea is to ask if a token has undergone a semantic changepoint along a set time period.
# A changepoint here is defined as a token attaining a new association or "meaning" as scientists make new discoveries.
# An iconic example of this would be the word pandemic.
# Pandemic was associated with H1N1 because a lot of papers were talking about H1N1 during the late 2010s.
# RECENTLY people and scientists have begun to talk more about covid-19.
# This shift of focus is considered a "changepoint".

# +
from functools import partial
from pathlib import Path
import pickle
import re

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import plotnine as p9
import plydata as ply
import plydata.tidy as ply_tdy
import tqdm

import bayesian_changepoint_detection.offline_changepoint_detection as offcd
from detecta import detect_cusum
# -

# # Grab Word Frequencies

if not Path("output/all_tok_frequencies.tsv.xz").exists():
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
                    "tok": re.escape(tok),
                    "word_count": wv_model.wv.get_vecattr(tok, "count"),
                    "year": re.search(r"(\d+)_0", str(model)).group(1),
                }
                for tok in wv_model.wv.index_to_key
            ]
        )

        total_count = (word_freq_df >> ply.pull("word_count")).sum()

        word_freq_df = word_freq_df >> ply.define(frequency=f"word_count/{total_count}")

        data.append(word_freq_df)

# +
if not Path("output/all_tok_frequencies.tsv.xz").exists():
    all_word_freq_df = pd.concat(data) >> ply.call(".dropna")
    (
        all_word_freq_df
        >> ply.call(".dropna")
        >> ply.query("year != ''")
        >> ply.call(".astype", {"year": int, "frequency": float, "word_count": int})
        >> ply.call(
            ".to_csv",
            "output/all_tok_frequencies.tsv.xz",
            sep="\t",
            index=False,
            compression="xz",
        )
    )

else:
    all_word_freq_df = (
        pd.read_csv("output/all_tok_frequencies.tsv.xz", sep="\t", na_filter=False)
        >> ply.call(".dropna")
        >> ply.query("year != ''")
        >> ply.call(".astype", {"year": int, "frequency": float, "word_count": int})
    )

print((all_word_freq_df >> ply.select("tok") >> ply.distinct()).shape)
all_word_freq_df
# -

# ## Filter out Tokens

# plydata implicitly groups datapoints
# need to revert to pandas to actually cycle through each group
token_group = all_word_freq_df >> ply.call(".groupby", "tok")

if not Path("output/token_frequency_dict.pkl").exists():
    all_token_map = dict()
    for tok, group in tqdm.tqdm(token_group):
        all_token_map[tok] = group >> ply.arrange("year") >> ply.pull("year")

    pickle.dump(all_token_map, open("output/token_frequency_dict.pkl", "wb"))
else:
    all_token_map = pickle.load(open("output/token_frequency_dict.pkl", "rb"))

cleared_token_map = {
    tok: all_token_map[tok]
    for tok in tqdm.tqdm(all_token_map)
    if len(all_token_map[tok]) > 1
    and np.diff(all_token_map[tok]).sum() == len(all_token_map[tok]) - 1
}
print(len(cleared_token_map))

# ## Calculate frequency ratio

all_word_frequency_ratio_df = (
    all_word_freq_df
    >> ply.query(f"tok in {list(cleared_token_map.keys())}")
    >> ply.group_by("tok")
    >> ply.arrange("year")
    >> ply.define(
        frequency_ratio=lambda x: x.frequency / x.shift(1).frequency,
    )
    >> ply.ungroup()
    >> ply.call(".dropna")
    >> ply.define(year=lambda x: x["year"].apply(lambda y: f"{y-1}-{y}"))
)
print((all_word_frequency_ratio_df >> ply.select("tok") >> ply.distinct()).shape)
all_word_frequency_ratio_df.head()

# # Grab semantic change values

distance_files = list(
    Path("../multi_model_experiment/output/combined_inter_intra_distances").rglob(
        "saved_*_distance.tsv"
    )
)
print(len(distance_files))

year_distance_map = {
    re.search(r"\d+", str(year_file)).group(0): (
        pd.read_csv(str(year_file), sep="\t", na_filter=False)
    )
    for year_file in tqdm.tqdm(distance_files)
}

full_token_set_df = (
    pd.concat([year_distance_map[year] for year in tqdm.tqdm(year_distance_map)])
    >> ply.define(tok=lambda x: x.tok.apply(lambda y: re.escape(y)))
    >> ply_tdy.unite("year", "year_1", "year_2", sep="-")
)
print((full_token_set_df >> ply.select("tok") >> ply.distinct()).shape)
full_token_set_df.head()

freq_map = dict()
for tok, group in tqdm.tqdm(full_token_set_df.groupby("tok")):
    freq_map[tok] = group >> ply.arrange("year") >> ply.pull("ratio_metric")

# # Combine both information

# Using an idea similar to SCAF \[[1](https://doi.org/10.1007/s00799-019-00271-6)\], I'm combining the global qst metric with the percent change in frequency.
# SCAF uses percent change for both metrics; however, the caveat is that their method loses information for the first two timepoints.
# In my case given that global_distance_qst is a metric that's bound between 0 and 1 frequency percent change can be used directly.
# By combining these two terms I can use this metric as a means to estimate change and allow for bayesian changepoint detection to calculate the probability of a timepoint change.

data_rows = list()
for tok, group in tqdm.tqdm(all_word_frequency_ratio_df.groupby("tok")):
    if tok in freq_map and tok != "null":
        data_rows.append(
            {
                "ratio_metric": (freq_map[tok]),
                "frequency_ratio": (
                    group >> ply.arrange("year") >> ply.pull("frequency_ratio")
                ),
                "year": (group >> ply.arrange("year") >> ply.pull("year")),
                "tok": [tok]
                * (group >> ply.arrange("year") >> ply.pull("year")).shape[0],
            }
        )
print(len(data_rows))

final_token_df = pd.concat([pd.DataFrame.from_dict(data) for data in data_rows])
final_token_df

merged_frequency_df = (
    final_token_df
    >> ply.rename(year_pair="year")
    >> ply.select(
        "tok",
        "ratio_metric",
        "year_pair",
        "frequency_ratio",
    )
)
print((merged_frequency_df >> ply.select("tok") >> ply.distinct()).shape)
merged_frequency_df

change_metric_df = (
    merged_frequency_df
    >> ply.group_by("tok")
    >> ply.arrange("year_pair")
    >> ply.define(
        change_metric_ratio="ratio_metric + frequency_ratio",
    )
    >> ply.ungroup()
    >> ply.select(
        "year_pair", "tok", "ratio_metric", "frequency_ratio", "change_metric_ratio"
    )
)
change_metric_df >> ply.call(
    ".to_csv", "output/change_metric_abstracts.tsv", sep="\t", index=False
)
change_metric_df

# # Change point detection

# ## Bayesian

# Use Semantic Change Analysis with Frequency (SCAF) to perform bayesian change point detection.

if not Path("output/bayesian_changepoint_data_abstracts.tsv").exists():
    change_point_results = []
    for tok, tok_series_df in tqdm.tqdm(change_metric_df.groupby("tok")):

        change_metric_ratio = tok_series_df >> ply.pull("change_metric_ratio")
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
                        >> ply.call(".shift", -1)
                        >> ply.call(".dropna")
                        >> ply.pull("year_pair")
                    ),
                }
            )
        )

if not Path("output/bayesian_changepoint_data_abstracts.tsv", sep="\t").exists():
    change_point_df = pd.concat(change_point_results)
    change_point_df.to_csv(
        "output/bayesian_changepoint_data_abstracts.tsv", sep="\t", index=False
    )
else:
    change_point_df = pd.read_csv(
        "output/bayesian_changepoint_data_abstracts.tsv", sep="\t"
    )
change_point_df.head()

(change_point_df >> ply.arrange("-changepoint_prob") >> ply.slice_rows(30))

# ## CUSUM

# This section uses the CUSUM algorithm to determine changepoint events from time series. This algorithm uses a threshold value to determine cutoffs for a changepoint event. To figure out this value I'm using a cutoff of 2 standard deviations from the mean. The mean is calculated after filtering out one outlier that contains a value of 5000 when the other values are 700 and less. This allows for more potential matches to be found.

cutoff_df = (
    change_metric_df
    >> ply.query("change_metric_ratio < 1000")
    >> ply.select("change_metric_ratio")
    >> ply.call(".describe")
)
cutoff_df

threshold = cutoff_df.loc["mean"] * cutoff_df.loc["std"] * 2
threshold[0]

if not Path("output/cusum_changepoint_abstracts.tsv").exists():
    change_point_results = []
    for tok, tok_series_df in tqdm.tqdm(change_metric_df.groupby("tok")):

        change_metric_ratio = tok_series_df >> ply.pull("change_metric_ratio")
        year_series = tok_series_df >> ply.pull("year_pair")

        alarm, start, end, amp = detect_cusum(
            change_metric_ratio,
            threshold=threshold[0],
            drift=0.5,
            ending=True,
            show=False,
        )

        for values in zip(alarm, start, end, amp):

            change_point_results.append(
                {
                    "tok": tok,
                    "changepoint_idx": year_series[values[0]],
                    "start_idx": year_series[values[1]],
                    "end_idx": year_series[values[2]],
                    "value": values[3],
                }
            )

if not Path("output/cusum_changepoint_abstracts.tsv").exists():
    change_point_df = pd.DataFrame.from_records(change_point_results)
    change_point_df >> ply.call(
        ".to_csv", "output/cusum_changepoint_abstracts.tsv", sep="\t", index=False
    )
else:
    change_point_df = pd.read_csv("output/cusum_changepoint_abstracts.tsv", sep="\t")
print((change_point_df >> ply.select("tok") >> ply.distinct()).shape)
change_point_df

(change_point_df >> ply.arrange("-value") >> ply.slice_rows(30))

# # Take Home Points

# 1. Bayesian change point detection provides insight on the specific year period a semantic change point may have occurred.
# 2. Best positive result is pandemic which underwent a focus shift from bird flu and influenza to coronavirus.
# 3. Follow up analysis which will appear in the next notebook will involve looking at the top X token neighbors to the query word. By doing that one can estimate which kind of shift a word has undergone.
