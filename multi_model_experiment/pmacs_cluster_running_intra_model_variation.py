import copy
import csv
import itertools
import math
from pathlib import Path
import random

from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import Vocab
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import tqdm


def calculate_distances(
    first_model, second_model, neighbors: int = 25, year_pair: str = "year_pair_here"
):
    with open(f"output/temp/{year_pair}.tsv", "w") as outfile:

        writer = csv.DictWriter(
            outfile,
            delimiter="\t",
            fieldnames=["tok", "global_distance", "local_distance", "year_pair"],
        )
        writer.writeheader()

        common_vocab = set(first_model.vocab.keys()) & set(second_model.vocab.keys())
        for idx, token in enumerate(common_vocab):
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

            if idx % 10000 == 0 and idx > 0:
                outfile.flush()

    return []


if __name__ == "__main__":

    temp_output_path = "output/aligned_vectors_tmp"
    aligned_vectors = list(Path(f"{str(temp_output_path)}").rglob("*kv"))
    aligned_vector_map = dict()
    for vec in aligned_vectors:

        year = vec.stem.split("_")[0]

        if year not in aligned_vector_map:
            aligned_vector_map[year] = list()

        aligned_vector_map[year].append(vec)

    for year in aligned_vector_map:
        filename = f"output/intra_models/combined_{year}tsv"

        if Path(filename).exists():
            continue

        year_mapper = list(itertools.combinations(list(aligned_vector_map[year]), 2))
        with Parallel(n_jobs=len(year_mapper), prefer="threads") as parallel:
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
                for aligned_vector in tqdm.tqdm(year_mapper)
            )

        # Merge the files into one
        all_temp_files = list(Path("output/temp").rglob(f"{year}*{year}*tsv"))
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
