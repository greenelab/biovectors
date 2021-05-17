import itertools
from multiprocessing import Process, JoinableQueue, Manager
from typing import Any, Mapping, Sequence, Tuple, Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import orthogonal_procrustes
import tqdm
from umap.parametric_umap import ParametricUMAP


def generate_timeline(
    year_comparison_dict: Mapping[str, pd.DataFrame], query_token: str = "be"
) -> pd.DataFrame:
    """
    This function is designed to generate a timeline of the query token.
    The timeline is basically extracting a particular row from the year comparison dataframe.

    Parameters:
        year_comparison_dict - Dictionary containing year to year comparisons
        query_token - the query token in question
    """
    timeline_df = pd.DataFrame(
        [], columns=["token", "global_dist", "local_dist", "shift", "year_label"]
    )
    for year_comparison in year_comparison_dict:
        timeline_df = timeline_df.append(
            year_comparison_dict[year_comparison]
            .assign(year_label=year_comparison)
            .query(f"token == '{query_token}'")
        )

    return timeline_df


def _calculate_distances(
    year_one_model: pd.DataFrame,
    year_two_model: pd.DataFrame,
    neighbors: int,
    input_queue: JoinableQueue,
    output_queue: JoinableQueue,
):

    while True:
        tok = input_queue.get()

        if tok is None:
            break

        w_t1 = (
            year_one_model.query(f"token == {repr(tok)}").drop("token", axis=1).values
        )
        w_t2 = (
            year_two_model.query(f"token == {repr(tok)}").drop("token", axis=1).values
        )

        model_one_neighbors = list(
            map(lambda x: x[0], get_neighbors(year_one_model, tok, neighbors))
        )
        model_two_neighbors = list(
            map(lambda x: x[0], get_neighbors(year_two_model, tok, neighbors))
        )

        # w_(t) and w_(t+1)
        word_vec = np.vstack([w_t1, w_t2])

        # N(w_(t)) union N(w_(t+1))
        neighbor_vec = np.vstack(
            [
                year_one_model.query(f"token in {model_one_neighbors}")
                .set_index("token")
                .values,
                year_two_model.query(f"token in {model_two_neighbors}")
                .set_index("token")
                .values,
            ]
        )

        # Grab the similarity of word vectors
        output_queue.put(
            {
                "token": tok,
                "local_dist": pdist(
                    1 - cdist(word_vec, neighbor_vec, "cosine"), "cosine"
                ).item(),
                "global_dist": pdist(word_vec, "cosine").item(),
            }
        )

    # Poison pill to end the parallelization
    output_queue.put(None)


def get_global_local_distance(
    year_one_model: pd.DataFrame,
    year_two_model: pd.DataFrame,
    shared_tokens: Sequence[str],
    neighbors: int = 5,
    n_jobs: int = 3,
) -> pd.DataFrame:
    """
    This function is designed to get the local distance of ever token between two years.
    Local distance is defined as the cosine similarity of a token's neighbor's similarity:
        Cossim(query_token_year_one_similarity, query_token_year_two_similarity)
        query_token_year_one_similarity ~ cossim(query_token_year_one, all_token_neighbors_in_both_years)
        query_token_year_two_similarity ~ cossim(query_token_year_two, all_token_neighbors_in_both_years)

    Parameters:
        year_one_model - The first year to be compared
        year_two_model - the second year to be compared
        shared_tokens - the list of all tokens shared across the years
        neighbors - the number of neighbor tokens to use
    """
    token_distance = []
    counter = 0
    with Manager() as m:
        tok_queue = m.JoinableQueue()
        dist_queue = m.JoinableQueue()

        runnable_jobs = []
        for job in range(n_jobs):
            p = Process(
                target=_calculate_distances,
                args=(year_one_model, year_two_model, neighbors, tok_queue, dist_queue),
            )
            runnable_jobs.append(p)
            p.start()

        for idx, tok in enumerate(shared_tokens):
            tok_queue.put(tok)

        # Poison pill to break out the parallel pipeline
        for job in range(n_jobs):
            tok_queue.put(None)

        while True:

            dist = dist_queue.get()
            if dist is None:
                counter += 1

                if counter == n_jobs:
                    break

                continue

            token_distance.append(dist)

    total_distance = pd.DataFrame.from_records(token_distance)

    return total_distance


def get_neighbors(
    word_vector_matrix: pd.DataFrame,
    query_token: str,
    neighbors: int = 10,
) -> Sequence[Tuple[np.array, np.array]]:
    """
    This function is designed to get X neighbors from a word_vector matrix as well as
    the query token.

    Parameters:
        word_vector_matrix - matrix of word vectors (tokens x their dimensions)
        query_token - the token to be comapred
        neighbors - the number of neighbor tokens to use
    """
    if neighbors < 0:
        raise RuntimeError("Number of neighbors cannot be negative.")

    if neighbors == 0:
        return []

    else:
        word_vector_matrix = word_vector_matrix.sort_values("token")

        sim_mat = 1 - cdist(
            word_vector_matrix.query(f"token == {repr(query_token)}")
            .drop("token", axis=1)
            .values,
            word_vector_matrix.drop("token", axis=1).values,
            "cosine",
        )
        # sim_mat = 1 - sim_mat
        token_neighbors = sim_mat[0, :].argsort()[-neighbors - 1 :][::-1]

        return list(
            zip(
                word_vector_matrix.iloc[token_neighbors[1:]].token.tolist(),
                sim_mat[0, :][token_neighbors[1:]],
            )
        )


def project_token_timeline(
    token: str,
    aligned_models: Mapping[str, Any],
    model: ParametricUMAP,
    neighbors: int = 0,
) -> pd.DataFrame:
    """
    This function is designed to project a query vector across all years onto a UMAP plot.

    Parameters:
        token - the token to be projected
        aligned_models - the word2vec model's matrix that has been aligned to the most recent year
        model - the trained TSNE model to project the tokens on
        neighbors - the number of neighbor tokens to gather
    """

    coordinates = []
    for year in aligned_models:

        if token not in aligned_models[year].token.tolist():
            continue

        projected_coord = model.transform(
            aligned_models[year].query(f"token == '{token}'").set_index("token").values
        )

        coordinates.append(
            {
                "umap_dim1": projected_coord[0][0],
                "umap_dim2": projected_coord[0][1],
                "year": year,
                "token": token,
                "label": "main",
            }
        )

        neighbor_list = get_neighbors(
            aligned_models[year],
            token,
            neighbors=neighbors,
        )

        for neighbor in neighbor_list:
            projected_coord = model.transform(
                aligned_models[year]
                .query(f"token == {repr(neighbor[0])}")
                .set_index("token")
                .values
            )

            coordinates.append(
                {
                    "umap_dim1": projected_coord[0][0],
                    "umap_dim2": projected_coord[0][1],
                    "year": year,
                    "token": neighbor[0],
                    "label": "neighbor",
                }
            )

    return pd.DataFrame.from_records(coordinates)


def window(seq: Sequence[Any], n: int = 2) -> Iterable[Any]:
    """
    Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    Parameters:
        Any iterable
    """
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
