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

# # Confirm Novel Distance Metric Corrects for Variance

# +
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pandas as pd
import plotnine as p9
import plydata as ply
import plydata.tidy as ply_tdy
from scipy.spatial.distance import cdist
import tqdm

from biovectors_modules.word2vec_analysis_helper import window
from biovectors_modules.plot_helper import overlay_token_with_model
# -

# # Grab Average Distance for Single Model

year_models = sorted(
    list(Path("output/aligned_models").glob("*model/*_0_fulltext_aligned.kv")),
    key=lambda x: int(x.stem.split("_")[0]),
)

common_vocab = set()
for model in tqdm.tqdm(reversed(year_models)):
    word_model = KeyedVectors.load(str(model))
    if len(common_vocab) == 0:
        common_vocab = set(word_model.vocab.keys())
    else:
        common_vocab &= set(word_model.vocab.keys())

# +
combined_model_data_rows = []
for model_one, model_two in window(year_models, 2):
    model_one_prefix = model_one.stem.split("_")[0]
    model_two_prefix = model_two.stem.split("_")[0]

    word2vec_model_one = KeyedVectors.load(str(model_one))
    word2vec_model_two = KeyedVectors.load(str(model_two))

    for tok in tqdm.tqdm(common_vocab):
        cosine_dist_values = cdist(
            [word2vec_model_one[tok]], [word2vec_model_two[tok]], "cosine"
        )
        combined_model_data_rows.append(
            {
                "tok": tok,
                "distance": cosine_dist_values[0][0],
                "year_1": model_one_prefix,
                "year_2": model_two_prefix,
            }
        )

single_model_df = pd.DataFrame.from_records(combined_model_data_rows)
print(single_model_df.shape)
single_model_df

# +
single_distance_df = (
    single_model_df
    >> ply_tdy.unite("timepoint", "year_1", "year_2", sep="-")
    >> ply.group_by("timepoint")
    >> ply.define(distance="mean(distance)")
    >> ply.ungroup()
    >> ply.select("timepoint", "distance")
    >> ply.distinct()
    >> ply.arrange("timepoint")
)

middle_estimate_2010 = single_distance_df.iloc[10].values[1]
middle_estimate_2015 = single_distance_df.iloc[15].values[1]

single_distance_df = single_distance_df >> ply.define(
    pct_diff_2010="abs(distance/middle_estimate_2010 - 1)",
    pct_diff_2015="abs(distance/middle_estimate_2015 - 1)",
)
single_distance_df
# -

# # Grab Average Distance for Multi Models

# +
multi_model_distances = sorted(
    list(Path("output/final_distances/pubtator").glob("*tsv")),
    key=lambda x: int(x.stem.split("_")[1].split("-")[0]),
)

multi_model_df_list = []
for multi_distance_path in tqdm.tqdm(multi_model_distances):
    years_compared = multi_distance_path.stem.split("_")[1].split("-")
    temp_multi_model_df = (
        pd.read_csv(multi_distance_path, sep="\t")
        >> ply.define(
            intra_distance=f"averaged_{years_compared[0]}_distance+averaged_{years_compared[1]}_distance"
        )
        >> ply.rename(
            distance="adjusted_distance",
            timepoint="year_pair",
            inter_distance="averaged_inter_distance",
        )
        >> ply.select(
            "tok", "distance", "timepoint", "inter_distance", "intra_distance"
        )
    )
    multi_model_df_list.append(temp_multi_model_df)
multi_model_df = pd.concat(multi_model_df_list, axis=0)
multi_model_df
# -

multi_distance_df = (
    multi_model_df
    >> ply.query(f"tok in {list(common_vocab)}")
    >> ply.group_by("timepoint")
    >> ply.define(distance="mean(distance)")
    >> ply.ungroup()
    >> ply.select("timepoint", "distance")
    >> ply.distinct()
    >> ply.arrange("timepoint")
)
middle_estimate_2010 = multi_distance_df.iloc[10].values[1]
middle_estimate_2015 = multi_distance_df.iloc[15].values[1]
multi_distance_df = multi_distance_df >> ply.define(
    pct_diff_2015="abs(distance/middle_estimate_2015 - 1)",
    pct_diff_2010="abs(distance/middle_estimate_2010 - 1)",
)
multi_distance_df

output_file_folder = Path("output/figure_data_and_figures/novel_distance_visualization")

g = (
    single_distance_df
    >> ply.define(label='"single_model"')
    >> ply.call(".append", multi_distance_df >> ply.define(label='"correction_model"'))
    >> (
        p9.ggplot()
        + p9.aes(x="timepoint", y="pct_diff_2010", color="label", group="label")
        + p9.geom_point()
        + p9.geom_line()
        + p9.coord_flip()
        + p9.theme_seaborn(style="white")
        + p9.labs(
            title="Percent Difference Relative to 2010-2011",
            x="Time Periods",
            y="Percent Difference",
        )
        + p9.scale_color_brewer("qual", palette="Dark2")
    )
)
print(g)
g.save(f"{str(output_file_folder)}/percent_difference_2010-2011_plot.svg")
g.save(f"{str(output_file_folder)}/percent_difference_2010-2011_plot.png", dpi=300)

g = (
    single_distance_df
    >> ply.define(label='"single_model"')
    >> ply.call(".append", multi_distance_df >> ply.define(label='"correction_model"'))
    >> (
        p9.ggplot()
        + p9.aes(x="timepoint", y="pct_diff_2015", color="label", group="label")
        + p9.geom_point()
        + p9.geom_line()
        + p9.coord_flip()
        + p9.theme_seaborn(style="white")
        + p9.labs(
            title="Percent Difference Relative to 2015-2016",
            x="Time Periods",
            y="Percent Difference",
        )
        + p9.scale_color_brewer("qual", palette="Dark2")
    )
)
print(g)
g.save(f"{str(output_file_folder)}/percent_difference_2015-2016_plot.svg")
g.save(f"{str(output_file_folder)}/percent_difference_2015-2016_plot.png", dpi=300)

(
    single_distance_df
    >> ply.define(label='"single_model"')
    >> ply.call(".append", multi_distance_df >> ply.define(label='"correction_model"'))
    >> ply.call(
        ".to_csv",
        f"{str(output_file_folder)}/correction_metric_data.tsv",
        sep="\t",
        index=False,
    )
)

# # Plot known changepoint examples.

tok = ["pandemic", "cas9"]
g = overlay_token_with_model(tok, multi_model_df, multi_distance_df)
print(g)
g.save(f'{str(output_file_folder)}/{"_".join(tok)}_changepoint_example_plot.svg')
g.save(
    f'{str(output_file_folder)}/{"_".join(tok)}_changepoint_example_plot.png', dpi=300
)
