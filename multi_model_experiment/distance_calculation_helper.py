import csv
from pathlib import Path

import gensim
from gensim.models import KeyedVectors
import numpy as np
import tqdm


def calculate_distances(
    first_model,
    second_model,
    subset_tokens: set = {},
    neighbors: int = 25,
    year_pair: str = "year_pair_here",
):
    """
    This function is designed to calculate the local and global distance metric for word vectors
    Global distance is defined as the cosine distance between words in year with their second year counterparts
    Local distnace is defined as the cosine distance of a word's similarity to its neighbors across time

    Arguments:
        first_model - the first aligned word vector model
        sectond model - the second aligned word vector model
        neighbors - the number of neighbors to use in the local modle comparison
        year_pair - the two years that will be compared
    """
    if Path(f"output/temp/{year_pair}.tsv").exists():
        return []

    # Have to write to file as RAM cannot hold all word pairs
    with open(f"output/temp/{year_pair}.tsv", "w") as outfile:

        writer = csv.DictWriter(
            outfile,
            delimiter="\t",
            fieldnames=["tok", "global_distance", "local_distance", "year_pair"],
        )

        writer.writeheader()

        # Gensim make large changes in >4.0
        # PMACS uses 3.8.3 so this version check is necessary
        if gensim.__version == "3.8.3":
            if len(subset_tokens) > 0:
                common_vocab = (
                    set(first_model.vocab.keys())
                    & set(second_model.vocab.keys())
                    & subset_tokens
                )
            else:
                common_vocab = set(first_model.vocab.keys()) & set(
                    second_model.vocab.keys()
                )
        else:
            if len(subset_tokens) > 0:
                common_vocab = (
                    set(first_model.vocab.keys())
                    & set(second_model.vocab.keys())
                    & subset_tokens
                )
            else:
                common_vocab = set(first_model.vocab.keys()) & set(
                    second_model.vocab.keys()
                )

        for token in tqdm.tqdm(common_vocab):
            first_model_neighbors, first_model_sims = zip(
                *first_model.most_similar(token, topn=neighbors)
            )
            second_model_neighbors, second_model_sims = zip(
                *second_model.most_similar(token, topn=neighbors)
            )
            years_neighbors_union = np.vstack(
                [
                    first_model[first_model_neighbors],
                    second_model[second_model_neighbors],
                ]
            )
            first_neighborhood_sims = KeyedVectors.cosine_similarities(
                first_model[token], years_neighbors_union
            )
            second_neighborhood_sims = KeyedVectors.cosine_similarities(
                second_model[token], years_neighbors_union
            )

            writer.writerow(
                {
                    "tok": token,
                    "global_distance": 1
                    - (
                        KeyedVectors.cosine_similarities(
                            first_model[token], second_model[token][:, np.newaxis].T
                        ).item()
                    ),
                    "local_distance": 1
                    - (
                        KeyedVectors.cosine_similarities(
                            first_neighborhood_sims,
                            second_neighborhood_sims[:, np.newaxis].T,
                        ).item()
                    ),
                    "year_pair": year_pair,
                }
            )

    return []
