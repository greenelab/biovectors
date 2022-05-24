from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Iterable

from gensim.models import Word2Vec
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import plydata as ply
import tqdm
from wordcloud import WordCloud


def deidentify_concepts(concept: str, concept_mapper: dict):
    if concept.startswith("disease_") or concept.startswith("chemical_"):
        concept = concept[concept.find("_") + 1 :]

    return (
        f"{concept_mapper[concept]} ({concept})"
        if concept in concept_mapper
        else concept
    )


def generate_neighbor_table(
    word2vec_model_list: list,
    query_token: str,
    changepoint_df: pd.DataFrame,
    concept_mapper: dict,
    n_neighbors: int = 10,
    output_file_folder: Path = Path("output"),
    save_file: bool = False,
):
    neighbor_df = pd.DataFrame()
    for model in tqdm.tqdm(word2vec_model_list):
        word_model = Word2Vec.load(str(model))
        if query_token in word_model.wv.vocab:
            neighbors = list(
                map(
                    lambda x: deidentify_concepts(x[0], concept_mapper),
                    word_model.wv.most_similar(query_token, topn=n_neighbors),
                )
            )
            year = model.stem.split("_")[0]
            neighbor_df = neighbor_df >> ply.define((year, neighbors))

    print(changepoint_df >> ply.query(f"tok=='{query_token}'"))
    if save_file:
        changepoint = (
            changepoint_df
            >> ply.query(f"tok=='{query_token}'")
            >> ply.pull("changepoint_idx")
        )
        if len(changepoint) == 0:
            changepoint = ["no_changepoint_detected"]
        neighbor_df.T >> ply.call(
            ".to_csv",
            f"{str(output_file_folder)}/{query_token}_neighbors_{changepoint[0]}.tsv",
            sep="\t",
        )
    return neighbor_df


def overlay_token_with_model(
    token_list: list,
    single_token_dist_df: pd.DataFrame,
    multi_distance_df: pd.DataFrame,
) -> p9.ggplot:
    all_changepoints_df = pd.DataFrame()

    for query_tok in token_list:
        known_changepoint_df = (
            single_token_dist_df
            >> ply.query(f"tok in '{query_tok}'")
            >> ply.define(label=f'"{query_tok} token"')
            >> ply.select("tok", "label", "distance", "timepoint")
        )

        middle_estimate = known_changepoint_df.iloc[  # noqa:F841
            int(known_changepoint_df.shape[0] / 2)
        ]["distance"]

        known_changepoint_df = known_changepoint_df >> ply.define(
            pct_diff="abs(distance/middle_estimate -1)"
        )

        all_changepoints_df = all_changepoints_df >> ply.call(
            ".append", known_changepoint_df
        )

    multi_distance_df = (
        multi_distance_df
        >> ply.rename(pct_diff="pct_diff_2010")
        >> ply.select("timepoint", "distance", "pct_diff")
        >> ply.define(label='"correction_model"')
        >> ply.call(".append", all_changepoints_df)
        >> ply.select("-tok")
    )

    labels = [f"{tok} token" for tok in token_list]

    g = (
        multi_distance_df
        >> ply.define(
            label=pd.Categorical(
                multi_distance_df >> ply.pull("label"),
                categories=["correction_model"] + labels,
            )
        )
        >> (
            p9.ggplot()
            + p9.aes(x="timepoint", y="pct_diff", color="label", group="label")
            + p9.geom_point()
            + p9.geom_line()
            + p9.coord_flip()
            + p9.theme_seaborn(style="white")
            + p9.labs(
                title="Percent Difference Relative to 2010-2011",
                x="Time Periods",
                y="Percent Difference",
            )
            + p9.scale_color_brewer(type="qual", palette="Dark2")
        )
    )
    return g


def plot_local_global_distances(
    timeline_df: pd.DataFrame, token: str
) -> Tuple[p9.ggplot, p9.ggplot, p9.ggplot, p9.ggplot]:
    """
    This function is designed to create a line plot of a token's global and local distance.
    The global distance represents how the token vector itself is changing across the years,
    while the local distance represents how a token's similarity towards its neighbors
    change across the years.

    Parameters
        timeline_df - a panda dataframe containing tokens with local_distance, global_distance and the corresponding z-scores
    """
    timeline_df = timeline_df >> ply.query("token==@token")

    text_color = (
        timeline_df
        >> ply.define(
            text_color=ply.if_else("global_dist >= 0.7", "'black'", "'white'")
        )
        >> ply.pull("text_color")
    )

    global_distance_plot = (
        p9.ggplot(
            timeline_df, p9.aes(x="year_origin", y="year_compared", fill="global_dist")
        )
        + p9.geom_tile(p9.aes(width=1, height=1))
        + p9.geom_text(
            p9.aes(label="global_dist"), format_string="{:.2f}", color=text_color
        )
        + p9.labs(
            x="Year Start",
            y="Year End",
            title=f"Global Distance for '{token}'",
            fill="Global Dist",
        )
        + p9.theme_seaborn(style="ticks", context="paper")
        + p9.scale_fill_cmap(limits=[0, 1])
        + p9.theme(figure_size=(10, 8), text=p9.element_text(size=11))
    )

    text_color = (
        timeline_df
        >> ply.define(text_color=ply.if_else("local_dist >= 0.7", "'black'", "'white'"))
        >> ply.pull("text_color")
    )

    local_distance_plot = (
        p9.ggplot(
            timeline_df, p9.aes(x="year_origin", y="year_compared", fill="local_dist")
        )
        + p9.geom_tile(p9.aes(width=1, height=1))
        + p9.geom_text(
            p9.aes(label="local_dist"), format_string="{:.2f}", color=text_color
        )
        + p9.labs(
            x="Year Start",
            y="Year End",
            title=f"Local Distance for '{token}'",
            fill="Local Dist",
        )
        + p9.theme_seaborn(style="ticks", context="paper")
        + p9.scale_fill_cmap(limits=[0, 1])
        + p9.theme(figure_size=(10, 8), text=p9.element_text(size=11))
    )

    text_color = (
        timeline_df
        >> ply.define(
            text_color=ply.if_else("z_global_dist >= 2", "'black'", "'white'")
        )
        >> ply.pull("text_color")
    )

    z_global_distance_plot = (
        p9.ggplot(
            timeline_df,
            p9.aes(x="year_origin", y="year_compared", fill="z_global_dist"),
        )
        + p9.geom_tile(p9.aes(width=1, height=1))
        + p9.geom_text(
            p9.aes(label="z_global_dist"), format_string="{:.2f}", color=text_color
        )
        + p9.labs(
            x="Year Start",
            y="Year End",
            title=f"Z-score Global Distance for '{token}'",
            fill="Z(Global Dist)",
        )
        + p9.theme_seaborn(style="ticks", context="paper")
        + p9.scale_fill_cmap(limits=[-3, 3])
        + p9.theme(figure_size=(10, 8), text=p9.element_text(size=11))
    )

    text_color = (
        timeline_df
        >> ply.define(text_color=ply.if_else("z_local_dist >= 2", "'black'", "'white'"))
        >> ply.pull("text_color")
    )

    z_local_distance_plot = (
        p9.ggplot(
            timeline_df, p9.aes(x="year_origin", y="year_compared", fill="z_local_dist")
        )
        + p9.geom_tile(p9.aes(width=1, height=1))
        + p9.geom_text(
            p9.aes(label="z_local_dist"), format_string="{:.2f}", color=text_color
        )
        + p9.labs(
            x="Year Start",
            y="Year End",
            title=f"Z-score Local Distance for '{token}'",
            fill="Z(Local Dist)",
        )
        + p9.theme_seaborn(style="ticks", context="paper")
        + p9.scale_fill_cmap(limits=[-3, 3])
        + p9.theme(figure_size=(10, 8), text=p9.element_text(size=11))
    )

    return (
        global_distance_plot,
        local_distance_plot,
        z_global_distance_plot,
        z_local_distance_plot,
    )


def plot_token_timeline(timeline_df: pd.DataFrame) -> p9.ggplot:
    """
    This function is designed to plot a tsne projection of each word vector across the years.

    Parameters
        timeline_df - a panda dataframe containing the main token along with its neighbors
    """

    # Title of the plot
    token_title = timeline_df.query("label=='main'").iloc[0]["token"].upper()
    token_title = f"{token_title} Timeline"

    # Plotnine is awesome
    # Generate the plot
    g = (
        p9.ggplot(
            timeline_df.query("label=='main'"),
            p9.aes(x="umap_dim1", y="umap_dim2", label="year"),
        )
        + p9.geom_text(
            size=8,
            ha="left",
            va="baseline",
            position=p9.position_nudge(x=0.04, y=-0.01),
        )
        + p9.geom_point(size=1)
        + p9.theme(
            axis_line=p9.element_blank(),
            axis_line_x=p9.element_blank(),
            axis_line_y=p9.element_blank(),
            panel_grid=p9.element_blank(),
            panel_background=p9.element_rect(fill="white"),
        )
        + p9.labs(title=token_title)
        + p9.xlim(
            [
                np.floor(timeline_df.umap_dim1.min()),
                np.ceil(timeline_df.umap_dim1.max()),
            ]
        )
    )

    return g


def plot_wordcloud_neighbors(
    timeline_df: pd.DataFrame,
    num_plots_per_row: int = 3,
    plot_filename: str = "token_neighbors.png",
):
    """
    This function is designed to make a plot with each word cloud as a subplot.
    The issue here is the resolution can be wonky, but still important to have enabled.

    Parameters
        timeline_df - a panda dataframe containing the main token along with its neighbors
        num_plots_per_row - the number of subplots to have per row
        plot_filename - the name of the png file to be saved
    """

    # create the figures and sort the years
    plt.figure(figsize=(10, 10))
    years = timeline_df.sort_values("year").year.unique()
    token = timeline_df.iloc[0].token.upper()

    # Get the number of subplots per row
    num_plots = num_plots_per_row
    rows = int(np.ceil(len(years) / num_plots))

    # Plot the word clouds
    for idx, year in enumerate(years):
        neighbors = timeline_df.query(f"label=='neighbor'&year=='{year}'")
        token_text = dict(
            zip(
                neighbors.token.tolist(), [1 for idx in range(neighbors.token.shape[0])]
            )
        )

        i = idx + 1
        wc = WordCloud(
            background_color="white", color_func=lambda *args, **kwargs: (43, 140, 190)
        )
        wc.generate_from_frequencies(token_text)
        plt.plot()
        plt.subplot(rows, num_plots, i).set_title(year)
        plt.imshow(wc)
        plt.axis("off")
        plt.suptitle(f"{token} Neighbors", fontsize=20)

    plt.savefig(
        plot_filename, dpi=75, bbox_inches="tight", transparent="True", pad_inches=0
    )


def plot_wordcloud_neighbors_gif(
    timeline_df: pd.DataFrame,
    piece_folder: str = "output/gif_pieces",
    plot_filename: str = "token_neighbors.gif",
):
    """
    This function is designed to generate a gif of word clouds for each token analyzed in this project.
    Gif is desired as one doesnt have to worry about resolution issues compared to the
    making multiple subplots.

    Parameters
        timeline_df - a panda dataframe containing the main token along with its neighbors
        piece_folder - the folder to hold temporary files to make the gif
        plot_filename - the name of the gif file to be saved
    """
    # Create the folder
    Path(piece_folder).mkdir(parents=True, exist_ok=True)

    # Get the years to analyze
    years = timeline_df.sort_values("year").year.unique()

    # Plot the frames for the gif
    for idx, year in enumerate(years):
        neighbors = timeline_df.query(f"label=='neighbor'&year=='{year}'")
        token_text = dict(
            zip(
                neighbors.token.tolist(), [1 for idx in range(neighbors.token.shape[0])]
            )
        )

        wc = WordCloud(
            background_color="white", color_func=lambda *args, **kwargs: (43, 140, 190)
        )
        wc.generate_from_frequencies(token_text)
        plt.figure(figsize=(10, 10))
        plt.imshow(wc)
        plt.title(f"Year: {year}", fontsize=22)
        plt.axis("off")

        plt.savefig(
            f"{piece_folder}/{idx}.png",
            dpi=75,
            bbox_inches="tight",
            transparent="True",
            pad_inches=0,
        )
        plt.close()

    # Create the gif
    with imageio.get_writer(
        plot_filename, mode="I", format="GIF-FI", duration=3, quantizer="nq"
    ) as writer:
        gif_pieces = list(Path(piece_folder).rglob("*png"))
        for file in sorted(gif_pieces, key=lambda x: int(x.stem)):
            writer.append_data(imageio.imread(str(file)))
