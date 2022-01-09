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

# # Dive into time periods of change

# This notebook is designed to further examine individual changepoints that occurred in [01_change_point_detection.ipynb](01_change_point_detection.ipynb) and [02_changepoint_detection_comparison](02_changepoint_detection_comparison).

# +
from IPython import display
from pathlib import Path
import re

from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import pandas as pd
import plotnine as p9
import plydata as ply
import plydata.tidy as ply_tdy
from upsetplot import from_memberships, UpSet, from_indicators
import wordcloud
# -

# ## Load the changepoint predictions

change_point_df = pd.read_csv(
    "output/cusum_changepoint_abstracts.tsv", sep="\t"
) >> ply.define(tok=r'tok.str.replace("\-", "-", regex=False)')
change_point_df >> ply.slice_rows(10)

change_metric_df = pd.read_csv(
    "output/change_metric_abstracts.tsv", sep="\t", na_filter=False
)
change_metric_df >> ply.slice_rows(10)

# ## Load the word vector model and their indicies

word_models = list(Path("../multi_model_experiment/output/models").rglob("*/*model"))
word_models[0:2]

word_model_map = dict()
for word_model in word_models:
    match_obj = re.search(r"(\d+)_(\d).model", str(word_model))

    year = int(match_obj.group(1))
    if year not in word_model_map:
        word_model_map[year] = list()

    word_model_map[year].append(str(word_model))

word_model_loaded_map = {
    key: Word2Vec.load(sorted(word_model_map[key])[0]) for key in word_model_map
}

word_freq_count_cutoff = 30

word_model_cutoff_map = {
    key: {
        "model": word_model_loaded_map[key],
        "cutoff_index": min(
            map(
                lambda x: 999999
                if word_model_loaded_map[key].wv.get_vecattr(x[1], "count")
                > word_freq_count_cutoff
                else x[0],
                enumerate(word_model_loaded_map[key].wv.index_to_key),
            )
        ),
    }
    for key in word_model_loaded_map
}


# # Examine Handpicked Tokens with a high chance of Change

# Only used in this notebook
def examine_words_at_timepoint_range(
    word_model_map: dict, years_to_examine: list, tok: str = "the", topn: int = 25
):
    word_map = dict()
    for year in years_to_examine:
        vocab = list(word_model_map[year]["model"].wv.key_to_index.keys())
        if tok in vocab:
            word_neighbors = word_model_map[year]["model"].wv.most_similar(
                tok, topn=topn, clip_end=word_model_map[year]["cutoff_index"]
            )
            for neighbor in word_neighbors:
                if year not in word_map:
                    word_map[year] = list()

                word_map[year].append(neighbor[0])
    return word_map


def plot_values(
    metric_df: pd.DataFrame, changepoint_df: pd.DataFrame, tok: str = "the"
):
    metric_plot = (
        p9.ggplot(
            metric_df
            >> ply.query(f"tok=='{tok}'")
            >> ply_tdy.gather("metric", "value", ["ratio_metric", "frequency_ratio"])
        )
        + p9.aes(x="year_pair", y="value", group="metric", color="metric")
        + p9.geom_point()
        + p9.geom_line()
        + p9.coord_flip()
        + p9.scale_color_brewer(type="qual", palette=2)
        + p9.theme_seaborn("white")
        + p9.labs(
            x="Year Shift",
            y="Ratio",
            title=f"Frequency + Semantic Ratio for Token ('{tok}')",
        )
    )

    cusum_years_predicted = (
        changepoint_df >> ply.query(f"tok == '{tok}'") >> ply.pull("changepoint_idx")
    )

    for year in cusum_years_predicted:
        metric_plot += p9.annotate(
            "point",
            x=year,
            y=(
                metric_df
                >> ply.query(f"tok=='{tok}'")
                >> ply.query(f"year_pair=='{year}'")
                >> ply.pull("frequency_ratio")
            )[0],
            fill="red",
            size=4,
        )

        metric_plot += p9.annotate(
            "point",
            x=year,
            y=(
                metric_df
                >> ply.query(f"tok=='{tok}'")
                >> ply.query(f"year_pair=='{year}'")
                >> ply.pull("ratio_metric")
            )[0],
            fill="red",
            size=4,
        )

    return metric_plot


figure_output_path = Path("output/figures")

# ## Pandemic

# COVID-19. The drastic shift that came from the pandemic that was first talked about in 2019 but became more prevalent in 2020.

plot_values(change_metric_df, change_point_df, tok="pandemic")

token_map = examine_words_at_timepoint_range(
    word_model_cutoff_map, [2017, 2018, 2019, 2020], tok="pandemic", topn=10
)
token_map

tokens = from_memberships(token_map.values())
fig = plt.figure(figsize=(11, 8))
upset = UpSet(
    tokens,
    with_lines=False,
    show_counts=False,
    sort_by="cardinality",
    intersection_plot_elements=0,
)
upset.style_subsets(present="lockdown", facecolor="#7570b3", label="2020")
axes = UpSet.plot(upset, fig=fig)
axes["shading"].set_xlabel("Order 2017, 2018, 2019, 2020")
fig.savefig(f"{str(figure_output_path)}/pandemic_changepoint.svg")
fig.savefig(f"{str(figure_output_path)}/pandemic_changepoint.png", dpi=500)

# ## Rituximab

# Not sure the change here but seems like the time period is when the drug was being used to treat rheumatoid arthritis. More investigation is needed.

plot_values(change_metric_df, change_point_df, tok="abcc6")

token_map = examine_words_at_timepoint_range(
    word_model_cutoff_map,
    [2014, 2015, 2016, 2017, 2018],
    tok="abcc6",
    topn=10,
)

tokens = from_memberships(token_map.values())
fig = plt.figure(figsize=(11, 8))
upset = UpSet(
    tokens,
    with_lines=False,
    sort_by="cardinality",
    intersection_plot_elements=0,
)
upset.style_subsets(present="83-year", facecolor="#1b9e77", label="2005")
upset.style_subsets(present="mrp2", facecolor="#7570b3", label="2006")
axes = UpSet.plot(upset, fig=fig)
axes["shading"].set_xlabel("Order 2014, 2015, 2016, 2017, 2018")
fig.savefig(f"{str(figure_output_path)}/abcc6_changepoint.svg")
fig.savefig(f"{str(figure_output_path)}/abcc6_changepoint.png", dpi=500)

# ## Asthma

# There is semantic shift for asthma where in 2002-2003 seems like focus shifted from co-morbidity to drug treatments. Fun fact asthma is connected to type 2 diabetes (had no idea).

plot_values(change_metric_df, change_point_df, tok="asthmatics")

token_map = examine_words_at_timepoint_range(
    word_model_cutoff_map,
    [2003, 2004, 2005, 2006, 2007],
    tok="asthmatics",
    topn=5,
)
token_map

tokens = from_memberships(token_map.values())
fig = plt.figure(figsize=(11, 8))
upset = UpSet(
    tokens,
    with_lines=False,
    show_counts=False,
    sort_by="cardinality",
    intersection_plot_elements=0,
)
upset.style_subsets(present="newborn", facecolor="#1b9e77")
upset.style_subsets(present="balf", facecolor="#7570b3")
axes = UpSet.plot(upset, fig=fig)
axes["shading"].set_xlabel("Order 2003, 2004, 2005, 2006, 2007")
fig.savefig(f"{str(figure_output_path)}/asthmatics_changepoint.svg")
fig.savefig(f"{str(figure_output_path)}/asthmatics_changepoint.png", dpi=500)

# ## Atoms

# This shift seems to involve moving from analyzing DNA structure to more of an individual molecule focus.

plot_values(change_metric_df, change_point_df, tok="atoms")

token_map = examine_words_at_timepoint_range(
    word_model_cutoff_map,
    [2004, 2005, 2006, 2007, 2008, 2009],
    tok="atoms",
    topn=5,
)
token_map

tokens = from_memberships(token_map.values())
fig = plt.figure(figsize=(11, 8))
upset = UpSet(
    tokens,
    with_lines=False,
    show_counts=False,
    sort_by="cardinality",
    intersection_plot_elements=0,
)
upset.style_subsets(present="backbone", facecolor="#1b9e77")
upset.style_subsets(present="cations", facecolor="#7570b3")
axes = UpSet.plot(upset, fig=fig)
axes["shading"].set_xlabel("Order 2004, 2005, 2006, 2007, 2008, 2009")
fig.savefig(f"{str(figure_output_path)}/atoms_changepoint.svg")
fig.savefig(f"{str(figure_output_path)}/atoms_changepoint.png", dpi=500)

# ## Individual Year Changes

# 2001 token has a high chance of a shift moving from medical topics from cancer and healthcare to specifying time and dates.
# This leads to the question do other years have this short of shift?
# Turns out yes 2003 and 2004 have this shift but more data is needed in order to calculate the changepoint timeline (only used tokens present across all the years).

plot_values(change_metric_df, change_point_df, tok="2001")

token_map = examine_words_at_timepoint_range(
    word_model_cutoff_map,
    [2000, 2001, 2002, 2003, 2004],
    tok="2001",
    topn=5,
)

tokens = from_memberships(token_map.values())
fig = plt.figure(figsize=(11, 8))
upset = UpSet(
    tokens,
    with_lines=False,
    show_counts=False,
    sort_by="cardinality",
    intersection_plot_elements=0,
)
upset.style_subsets(present="campaign", facecolor="#1b9e77")
upset.style_subsets(present="july", facecolor="#7570b3")
axes = UpSet.plot(upset, fig=fig)
axes["shading"].set_xlabel("Order 2000, 2001, 2002, 2003, 2004")
fig.savefig(f"{str(figure_output_path)}/2001_changepoint.svg")
fig.savefig(f"{str(figure_output_path)}/2001_changepoint.png", dpi=500)

plot_values(change_metric_df, change_point_df, tok="2002")

token_map = examine_words_at_timepoint_range(
    word_model_cutoff_map,
    [2001, 2002, 2003, 2004, 2005],
    tok="2002",
    topn=5,
)
token_map

tokens = from_memberships(token_map.values())
fig = plt.figure(figsize=(11, 8))
upset = UpSet(
    tokens,
    with_lines=False,
    show_counts=False,
    sort_by="cardinality",
    intersection_plot_elements=0,
)
upset.style_subsets(present="british_journal_of_cancer", facecolor="#1b9e77")
upset.style_subsets(present="april", facecolor="#7570b3")
axes = UpSet.plot(upset, fig=fig)
axes["shading"].set_xlabel("Order 2001, 2002, 2003, 2004, 2005")
fig.savefig(f"{str(figure_output_path)}/2002_changepoint.svg")
fig.savefig(f"{str(figure_output_path)}/2002_changepoint.png", dpi=500)

plot_values(change_metric_df, change_point_df, tok="2005")

token_map = examine_words_at_timepoint_range(
    word_model_cutoff_map,
    [2002, 2003, 2004, 2005, 2006, 2007],
    tok="2005",
    topn=5,
)
token_map

tokens = from_memberships(token_map.values())
fig = plt.figure(figsize=(11, 8))
upset = UpSet(
    tokens,
    with_lines=False,
    show_counts=False,
    sort_by="cardinality",
    intersection_plot_elements=0,
)
upset.style_subsets(present="annual", facecolor="#1b9e77")
upset.style_subsets(present="usa", facecolor="#7570b3")
axes = UpSet.plot(upset, fig=fig)
axes["shading"].set_xlabel("Order 2001, 2002, 2003, 2004, 2005, 2006, 2007")
fig.savefig(f"{str(figure_output_path)}/2005_changepoint.svg")
fig.savefig(f"{str(figure_output_path)}/2005_changepoint.png", dpi=500)

plot_values(change_metric_df, change_point_df, tok="2010")

token_map = examine_words_at_timepoint_range(
    word_model_cutoff_map,
    [2006, 2007, 2008, 2009, 2010, 2011],
    tok="2010",
    topn=5,
)
token_map

tokens = from_memberships(token_map.values())
fig = plt.figure(figsize=(11, 8))
upset = UpSet(
    tokens,
    with_lines=False,
    show_counts=False,
    sort_by="cardinality",
    intersection_plot_elements=0,
)
upset.style_subsets(present="london", facecolor="#1b9e77")
upset.style_subsets(present="2011", facecolor="#7570b3")
axes = UpSet.plot(upset, fig=fig)
axes["shading"].set_xlabel("Order 2006, 2007, 2008, 2009, 2010, 2011")
fig.savefig(f"{str(figure_output_path)}/2005_changepoint.svg")
fig.savefig(f"{str(figure_output_path)}/2005_changepoint.png", dpi=500)

# # Take home points

# 1. Looks like my model is off by 1 for words such as corona virus and pandemic. This is probably due the way changepoint detection works and changes at the end of the series is hard to detect.
# 2. Asthma has an association with type 2 diabetes. Thats really interesting as I just hadn't fathomed there was a connection.
# 3. Years seem to undergo a shift in prevalent topics discussed to time indiciations. Need more data to be processed before I can truly say this trend is possible.
# 4. Unfortunately, majority of my findings only arise from an increase in frequency. This might change if I were to use full text rather than abstracts.
