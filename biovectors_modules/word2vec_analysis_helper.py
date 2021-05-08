import itertools
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


def get_global_distance(
    year_one_model: np.array, year_two_model: np.array, shared_tokens: Sequence[str]
) -> pd.DataFrame:
    """
    This function is designed to get the global distance of ever token between two years.
    Global distance is basically the cosine similarity (token_year_one, token_year_two)

    Parameters:
        year_one_model - The first year to be compared
        year_two_model - the second year to be compared
        shared_tokens - the list of all tokens shared across the years
    """

    distance_matrix = cdist(year_one_model, year_two_model, "cosine")
    token_dist_iterator = zip(shared_tokens, np.diag(distance_matrix))
    global_distance = pd.DataFrame.from_records(
        [{"token": token, "global_dist": dist} for token, dist, in token_dist_iterator]
    )

    return global_distance


def get_local_distance(
    year_one_model: np.array,
    year_two_model: np.array,
    shared_tokens: Sequence[str],
    neighbors: int = 5,
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

    # Grab the cosine sim for first year
    cosine_sim_matrix_one = 1 - pdist(year_one_model, "cosine")
    cosine_sim_matrix_one = squareform(cosine_sim_matrix_one, "tomatrix")

    # Grab the cosine sim for second year
    cosine_sim_matrix_two = 1 - pdist(year_two_model, "cosine")
    cosine_sim_matrix_two = squareform(cosine_sim_matrix_two, "tomatrix")

    token_distance = []
    n_rows = list(range(cosine_sim_matrix_one.shape[0]))

    for idx, row in enumerate(n_rows):
        # Sort and grab the neighbors from the similarity matrix
        token_neighbors_one = cosine_sim_matrix_one[row, :].argsort()[-neighbors:][::-1]
        token_neighbors_two = cosine_sim_matrix_two[row, :].argsort()[-neighbors:][::-1]

        # w_(t) and w_(t+1)
        word_vec = np.stack([year_one_model[idx, :], year_two_model[idx, :]])

        # N(w_(t)) union N(w_(t+1))
        neighbor_vec = np.vstack(
            [
                year_one_model[token_neighbors_one, :],
                year_two_model[token_neighbors_two, :],
            ]
        )

        # Grab the similarity of word vectors
        token_distance.append(
            pdist(1 - cdist(word_vec, neighbor_vec, "cosine"), "cosine").item()
        )

    local_distance = pd.DataFrame.from_records(
        [
            {"token": tok, "local_dist": dist}
            for tok, dist in zip(shared_tokens, token_distance)
        ]
    )

    return local_distance


def get_neighbors(
    word_vector_matrix: np.array,
    query_token: str,
    shared_tokens: list,
    neighbors: int = 10,
) -> Sequence[Tuple[np.array, np.array]]:
    """
    This function is designed to get X neighbors from a word_vector matrix as well as
    the query token.

    Parameters:
        word_vector_matrix - matrix of word vectors (tokens x their dimensions)
        query_token - the token to be comapred
        shared_tokens - the list of all tokens shared across the years
        neighbors - the number of neighbor tokens to use
    """
    if neighbors < 0:
        raise RuntimeError("Number of neighbors cannot be negative.")

    if neighbors == 0:
        return []

    else:
        word_vector_row = shared_tokens.index(query_token)
        sim_mat = squareform(1 - pdist(word_vector_matrix, "cosine"), "tomatrix")
        token_neighbors = sim_mat[word_vector_row, :].argsort()[-neighbors:][::-1]

        return list(
            zip(
                np.array(shared_tokens)[token_neighbors],
                sim_mat[word_vector_row, :][token_neighbors],
            )
        )


def project_token_timeline(
    token: str,
    aligned_models: Mapping[str, Any],
    model: ParametricUMAP,
    neighbors: int = 0,
) -> pd.DataFrame:
    """
    This function is designed to project a query vector across all years onto a tSNE plot.

    Parameters:
        token - the token to be projected
        aligned_models - the word2vec model's matrix that has been aligned to the most recent year
        model - the trained TSNE model to project the tokens on
        neighbors - the number of neighbor tokens to gather
    """

    main_token_index = aligned_models["shared_tokens"].index(token)
    coordinates = []
    for year in aligned_models:
        if year == "shared_tokens":
            continue

        projected_coord = model.transform(
            aligned_models[year][main_token_index : main_token_index + 1, :]
        )
        coordinates.append(
            {
                "umap_dim1": projected_coord[0][0],
                "umap_dim2": projected_coord[0][1],
                "year": year,
                "token": aligned_models["shared_tokens"][main_token_index],
                "label": "main",
            }
        )

        neighbor_list = get_neighbors(
            aligned_models[year],
            token,
            aligned_models["shared_tokens"],
            neighbors=neighbors,
        )

        for neighbor in neighbor_list:
            neighbor_token_index = aligned_models["shared_tokens"].index(neighbor[0])
            projected_coord = model.transform(
                aligned_models[year][neighbor_token_index : neighbor_token_index + 1, :]
            )

            coordinates.append(
                {
                    "umap_dim1": projected_coord[0][0],
                    "umap_dim2": projected_coord[0][1],
                    "year": year,
                    "token": aligned_models["shared_tokens"][neighbor_token_index],
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
