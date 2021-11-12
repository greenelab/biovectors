import copy
import csv
import itertools
import math
from pathlib import Path
import random

from gensim.models import KeyedVectors
from joblib import Parallel, delayed
import pandas as pd
import tqdm

from .distance_calculation_helper import calculate_distances

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
