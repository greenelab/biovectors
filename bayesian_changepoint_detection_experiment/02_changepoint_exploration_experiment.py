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

# This notebook is designed to further examine individual changepoints that occurred in [01_change_point_detection.ipynb](01_change_point_detection.ipynb).

# +
from IPython import display
from pathlib import Path
import re

from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import pandas as pd
import plotnine as p9
import plydata as ply
import wordcloud
# -

# ## Load the changepoint predictions

change_point_df = pd.read_csv("output/bayesian_changepoint_data.tsv", sep="\t")
change_point_df >> ply.slice_rows(10)

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

def examine_words_at_timepoint_range(
    word_model_map: dict, years_to_examine: list, tok: str = "the", topn: int = 25
):
    for year in years_to_examine:
        print(year)
        display.display(
            word_model_map[year]["model"].wv.most_similar(
                tok, topn=topn, clip_end=word_model_map[year]["cutoff_index"]
            )
        )
        print()


# ## Pandemic

# COVID-19. The drastic shift that came from the pandemic that was first talked about in 2019 but became more prevalent in 2020.

(
    p9.ggplot(change_point_df >> ply.query("tok=='pandemic'"))
    + p9.aes(x="year_pair", y="changepoint_prob", group=0)
    + p9.geom_point()
    + p9.geom_line()
    + p9.coord_flip()
    + p9.theme_seaborn("white")
    + p9.labs(
        x="years",
        y="Probability of Changepoint",
        title="Changepoint Prediction for Token ('pandemic')",
    )
)

examine_words_at_timepoint_range(
    word_model_cutoff_map, [2017, 2018, 2019, 2020], tok="pandemic", topn=20
)

# ## Rituximab

# Not sure the change here but seems like the time period is when the drug was being used to treat rheumatoid arthritis. More investigation is needed.

(
    p9.ggplot(change_point_df >> ply.query("tok=='rituximab'"))
    + p9.aes(x="year_pair", y="changepoint_prob", group=0)
    + p9.geom_point()
    + p9.geom_line()
    + p9.coord_flip()
    + p9.theme_seaborn("white")
    + p9.labs(
        x="years",
        y="Probability of Changepoint",
        title="Changepoint Prediction for Token ('rituximab')",
    )
)

examine_words_at_timepoint_range(
    word_model_cutoff_map,
    [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007],
    tok="rituximab",
    topn=20,
)

# ## Asthma

# There is semantic shift for asthma where in 2002-2003 seems like focus shifted from co-morbidity to drug treatments. Fun fact asthma is connected to type 2 diabetes (had no idea).

(
    p9.ggplot(change_point_df >> ply.query("tok=='asthmatics'"))
    + p9.aes(x="year_pair", y="changepoint_prob", group=0)
    + p9.geom_point()
    + p9.geom_line()
    + p9.coord_flip()
    + p9.theme_seaborn("white")
    + p9.labs(
        x="years",
        y="Probability of Changepoint",
        title="Changepoint Prediction for Token ('asthmatics')",
    )
)

examine_words_at_timepoint_range(
    word_model_cutoff_map,
    [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007],
    tok="asthmatics",
    topn=20,
)

# ## Individual Year Changes

# 2001 token has a high chance of a shift moving from medical topics from cancer and healthcare to specifying time and dates.
# This leads to the question do other years have this short of shift?
# Turns out yes 2003 and 2004 have this shift but more data is needed in order to calculate the changepoint timeline (only used tokens present across all the years).

(
    p9.ggplot(change_point_df >> ply.query("tok=='2001'"))
    + p9.aes(x="year_pair", y="changepoint_prob", group=0)
    + p9.geom_point()
    + p9.geom_line()
    + p9.coord_flip()
    + p9.theme_seaborn("white")
    + p9.labs(
        x="years",
        y="Probability of Changepoint",
        title="Changepoint Prediction for Token ('2001')",
    )
)

examine_words_at_timepoint_range(
    word_model_cutoff_map,
    [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007],
    tok="2001",
    topn=20,
)

examine_words_at_timepoint_range(
    word_model_cutoff_map,
    [2001, 2002, 2003, 2004, 2005, 2006, 2007],
    tok="2002",
    topn=20,
)

examine_words_at_timepoint_range(
    word_model_cutoff_map,
    [2001, 2002, 2003, 2004, 2005, 2006, 2007],
    tok="2003",
    topn=20,
)

# # Take home points

# 1. Looks like my model is off by 1 for words such as corona virus and pandemic. This is probably due the way changepoint detection works and changes at the end of the series is hard to detect.
# 2. Asthma has an association with type 2 diabetes. Thats really interesting as I just hadn't fathomed there was a connection.
# 3. Years seem to undergo a shift in prevalent topics discussed to time indiciations. Need more data to be processed before I can truly say this trend is possible.
# 4. More data to come ones the years have finally finished processing.
