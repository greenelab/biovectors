import copy
import csv
import itertools
import math
from pathlib import Path
import random

from gensim.models import Word2Vec, KeyedVectors
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import tqdm


def calculate_distances(
    first_model, second_model, neighbors: int = 25, year_pair: str = "year_pair_here"
):
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
        common_vocab = (
            set(first_model.key_to_index.keys())
            & set(second_model.key_to_index.keys())
            & subset_tokens
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


if __name__ == "__main__":
    tokens = pd.read_csv("output/subsetted_tokens.tsv", sep="\t")
    global subset_tokens
    subset_tokens = set(tokens.tok.tolist())

    temp_output_path = "output/aligned_vectors_tmp"
    aligned_vectors = list(Path(f"{str(temp_output_path)}").rglob("*kv"))
    aligned_vector_map = dict()
    for vec in aligned_vectors:

        year = vec.stem.split("_")[0]
        if year not in aligned_vector_map:
            aligned_vector_map[year] = list()

        aligned_vector_map[year].append(vec)

    all_years = list(aligned_vector_map.keys())
    years_to_parse = list(itertools.combinations(sorted(all_years), 2))

    for pair in years_to_parse:

        filename = f"output/inter_models/combined_{pair[0]}_{pair[1]}.tsv"
        if Path(filename).exists():
            continue

        year_mapper = list(
            itertools.product(aligned_vector_map[pair[0]], aligned_vector_map[pair[1]])
        )

        # Calculate the distances
        with Parallel(n_jobs=2, prefer="threads") as parallel:
            result = parallel(
                delayed(calculate_distances)(
                    KeyedVectors.load(str(aligned_vector[0])),
                    KeyedVectors.load(str(aligned_vector[1])),
                    year_pair=(
                        f"{aligned_vector[0].stem}-{aligned_vector[1].stem}"
                        if int(aligned_vector[0].stem.split("_")[0])
                        < int(aligned_vector[1].stem.split("_")[0])
                        else f"{aligned_vector[1].stem}-{aligned_vector[0].stem}"
                    ),
                )
                for aligned_vector in year_mapper
            )

        # Merge the files into one
        all_temp_files = list(Path("output/temp").rglob(f"{pair[0]}*{pair[1]}*tsv"))
        with open(filename, "w") as outfile:
            for idx, temp_file in tqdm.tqdm(enumerate(all_temp_files)):
                with open(str(temp_file), "r") as infile:
                    reader = csv.DictReader(infile, delimiter="\t")

                    if idx == 0:
                        writer = csv.DictWriter(
                            outfile, delimiter="\t", fieldnames=reader.fieldnames
                        )
                        writer.writeheader()

                    for row in reader:
                        writer.writerow(row)

        # Delete the temporary files
        for file in all_temp_files:
            file.unlink()
