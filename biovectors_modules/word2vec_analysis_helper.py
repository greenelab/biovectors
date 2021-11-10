import copy
import csv
import itertools
from multiprocessing import current_process, Process, JoinableQueue, Manager
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Iterable

from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import orthogonal_procrustes
import tqdm
from umap.parametric_umap import ParametricUMAP

QUEUE_SIZE = 500000  # Increase queue size


def align_word2vec_models(base_model: Word2Vec, model_to_align: Word2Vec) -> Word2Vec:
    """
    This function is designed to align word vectors onto the base word vector model.
    based on https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    """
    # Insure that the original object passed doesn't get changed
    # base_model = copy.deepcopy(base_model)
    model_to_align = copy.deepcopy(model_to_align)

    # Get shared vocabulary words
    base_model_vocab = set(base_model.wv.key_to_index.keys())
    model_to_align_vocab = set(model_to_align.wv.key_to_index.keys())

    # Sort based on frequency of both models
    common_vocab = base_model_vocab & model_to_align_vocab
    common_vocab = list(common_vocab)
    common_vocab.sort(
        key=lambda word: base_model.wv.get_vecattr(word, "count")
        + model_to_align.wv.get_vecattr(word, "count"),
        reverse=True,
    )

    # Resort word vectors based on frequency and
    # Replace the vectors themselves
    model_to_align_indicies = [
        model_to_align.wv.key_to_index[word] for word in common_vocab
    ]
    old_arr = model_to_align.wv.get_normed_vectors()
    new_arr = np.array([old_arr[index] for index in model_to_align_indicies])

    # Calculate orthogonal procrustes
    translation_matrix, _ = orthogonal_procrustes(new_arr, base_model.wv[common_vocab])
    model_to_align.wv.vectors = new_arr @ translation_matrix

    # Update the wordvector object to reflect new changes
    model_to_align.wv.index_to_key = common_vocab
    # old_vocab = model_to_align.wv.key_to_index
    model_to_align.wv.key_to_index = {
        word: new_index for new_index, word in enumerate(common_vocab)
    }

    # Return updated keyedvectors
    return model_to_align


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
    input_queue: JoinableQueue,
    global_word_matrix: np.array,
    year_indicies: dict,
    neighbors: int,
    outfile: str,
    part: int,
):
    """
    This function is designed to calculate the global and local distance of ever token between two years.

    Parameters:
        year_one_model - The first year to be compared
        year_two_model - the second year to be compared
        neighbors - the number of neighbor tokens to use
        input_queue - the multiprocessing queue that takes in the token to be analyzed
        output_queue - the multiprocessing queue that takes the analysis results for output later in the program
    """

    while Path(f"output/{Path(outfile).stem}_part{part.value}.tsv").exists():
        part.value += 1

    with open(f"output/{Path(outfile).stem}_part{part.value}.tsv", "w") as outfile:
        with tqdm.tqdm(
            desc=f"Process {part.value-1}", position=0, leave=True
        ) as progress_bar:
            writer = csv.DictWriter(
                outfile,
                delimiter="\t",
                fieldnames=["token", "global_dist", "local_dist", "year_pair"],
            )

            writer.writeheader()

            while True:
                token = input_queue.get()

                progress_bar.update(1)

                if token is None:
                    break

                token_indicies = list(map(lambda x: x[1], token[1]))
                global_distances = cdist(
                    global_word_matrix[token_indicies, :], global_word_matrix, "cosine"
                )

                # Calculate the similarity
                global_similarity = 1 - global_distances

                # Get year row
                year_occurence = list(map(lambda x: x[0], token[1]))

                # for each year comparison
                for year_pair in itertools.combinations(year_occurence, 2):

                    # First pair index
                    year_one_index = year_indicies[year_pair[0]]
                    row_one = year_occurence.index(year_pair[0])

                    # Second Pair Index
                    year_two_index = year_indicies[year_pair[1]]
                    row_two = year_occurence.index(year_pair[1])

                    # N(W_t))
                    year_one_year_one_sims = global_similarity[
                        row_one, range(*year_one_index)
                    ].argsort()[-neighbors - 1 :][::-1]

                    # N(W_t+1)
                    year_two_year_two_sims = global_similarity[
                        row_two, range(*year_two_index)
                    ].argsort()[-neighbors - 1 :][::-1]

                    # S(W_t) - [cossim(W_t, N(W_t)), cossim(W_t, N(W_t+1))]
                    s_t = np.hstack(
                        [
                            global_similarity[row_one, range(*year_one_index)][
                                year_one_year_one_sims[1:]
                            ],
                            global_similarity[row_one, range(*year_two_index)][
                                year_two_year_two_sims[1:]
                            ],
                        ]
                    )

                    # S(W_t+1) - [cossim(W_t+1, N(W_t)), cossim(W_t+1, N(W_t+1))]
                    s_t2 = np.hstack(
                        [
                            global_similarity[row_two, range(*year_one_index)][
                                year_one_year_one_sims[1:]
                            ],
                            global_similarity[row_two, range(*year_two_index)][
                                year_two_year_two_sims[1:]
                            ],
                        ]
                    )

                    # Global Distance
                    writer.writerow(
                        {
                            "token": token[0],
                            "global_dist": global_distances[
                                row_one, token_indicies[row_two]
                            ],  # cos-dist(w_t, w_t+1)
                            "local_dist": cdist(
                                s_t[np.newaxis, :], s_t2[np.newaxis, :], "cosine"
                            ).item(),  # cos-dist(s(W_t), s(W_t+1))
                            "year_pair": "-".join(year_pair),
                        }
                    )


def get_global_local_distance(
    global_word_matrix: np.array,
    occurrence_dict: dict,
    year_indicies: dict,
    neighbors: int = 25,
    n_jobs: int = 1,
    output_file: str = "all_distance_file.tsv",
):
    """
    This function is designed to get the global and local distance of ever token between two years.
    Local distance is defined as the cosine similarity of a token's neighbor's similarity:
        Cossim(query_token_year_one_similarity, query_token_year_two_similarity)
        query_token_year_one_similarity ~ cossim(query_token_year_one, all_token_neighbors_in_both_years)
        query_token_year_two_similarity ~ cossim(query_token_year_two, all_token_neighbors_in_both_years)

    Global distance is defined as cosine similarity of two tokens between two years.

    Parameters:
        global_word_matrix - combined word matricies
        occurrence_dict - a dictionary containing tokens as keys and list as values each value is a tuple with (year_occurred, word_matrix index)
        year_indicies - the start and stop index of every token in a word_matrix
        neighbors - the number of neighbor tokens to use for local distance
    """
    with Manager() as m:
        tok_queue = m.JoinableQueue(QUEUE_SIZE)
        part_counter = m.Value("i", 1)

        runnable_jobs = []
        for job in range(n_jobs):
            p = Process(
                target=_calculate_distances,
                args=(
                    tok_queue,
                    global_word_matrix,
                    year_indicies,
                    neighbors,
                    output_file,
                    part_counter,
                ),
            )
            runnable_jobs.append(p)
            p.start()

        token_iterator = sorted(
            list(occurrence_dict.items()), key=lambda x: len(x[1]), reverse=True
        )

        for token in token_iterator:

            # Make sure tokens have consecutive years
            # compare the differences between each year
            year_chain = sorted(list(map(lambda x: int(x[0]), token[1])))

            # Correct length should be array minus one
            year_len = len(year_chain) - 1

            # Make sure the years are one after another
            year_differences = sum(np.diff(year_chain) == 1)

            # Skip tokens only occurring in 2020 or tokens without consecutive years
            if len(year_chain) <= 1 or year_differences != year_len:
                continue

            tok_queue.put(token)

        # Poison pill to break out the parallel pipeline
        for job in range(n_jobs):
            tok_queue.put(None)

        # join the jobs
        for p in runnable_jobs:
            p.join()


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
        query_token_vector = (
            word_vector_matrix.query(f"token == {repr(query_token)}")
            .drop("token", axis=1)
            .values
        )

        remaining_token_vectors = word_vector_matrix.drop("token", axis=1).values
        sim_mat = 1 - cdist(query_token_vector, remaining_token_vectors, "cosine")

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
