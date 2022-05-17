import argparse
import csv
from pathlib import Path
import itertools
import multiprocessing
import sys

from gensim.models import Word2Vec, KeyedVectors
import tqdm

from word2vec_timeline_utils import (
    align_word2vec_models,
    calculate_distances,
    build_batch_filelist,
    merge_calculated_distances,
    window,
)


parser = argparse.ArgumentParser(
    description="Align word2vec models and calculate intra/inter year distances."
)
parser.add_argument(
    "--trained_models_folder",
    help="The folder that contains fully trained word2vec models grouped by year.",
)
parser.add_argument(
    "--aligned_folder", help="The folder to output aligned word2vec models."
)
parser.add_argument(
    "--intra_year_folder", help="The folder to output intra-year calculations."
)
parser.add_argument(
    "--inter_year_folder", help="The folder to output inter-year calculations."
)
parser.add_argument(
    "--base_model", help="Provides path to the base model for alignment."
)
parser.add_argument(
    "--max_processes",
    help="Specify the number of processes to use for distance calculation.",
    type=int,
    const=5,
    nargs="?",
)

argv = parser.parse_args()

MAX_PROCESSES = argv.max_processes
year_models = list(Path(argv.trained_models_folder).glob("*_model"))
years_to_iterate = sorted(year_models)

base_model = Word2Vec.load(argv.base_model)
for year_pair in window(years_to_iterate, 2):

    # Create Alginment Folders
    align_folder = Path(argv.aligned_folder)
    align_folder.mkdir(exist_ok=True, parents=True)

    temp_year_one_path = Path(f"{align_folder}/{year_pair[0].stem}")
    run_alignment_year_one = not temp_year_one_path.exists()
    temp_year_two_path = Path(f"{align_folder}/{year_pair[1].stem}")
    run_alignment_year_two = not temp_year_one_path.exists()

    # Create the Intra Year Folder
    intra_year_folder = Path(argv.intra_year_folder)
    intra_year_folder.mkdir(exist_ok=True, parents=True)

    intra_year_one_path = Path(
        f"{intra_year_folder}/calculated_{year_pair[0].stem.split('_')[0]}_intra_distances.tsv"
    )
    run_intra_year_one = not intra_year_one_path.exists()

    intra_year_two_path = Path(
        f"{intra_year_folder}/calculated_{year_pair[1].stem.split('_')[0]}_intra_distances.tsv"
    )
    run_intra_year_two = not intra_year_two_path.exists()

    # Create the Inter Year Folder
    inter_year_folder = Path(argv.inter_year_folder)
    intra_year_folder.mkdir(exist_ok=True, parents=True)
    inter_year_path = Path(
        f"{inter_year_folder}/calculated_{year_pair[0].stem.split('_')[0]}-{year_pair[1].stem.split('_')[0]}_inter_distances.tsv"
    )
    run_inter_year = not inter_year_path.exists()

    # Align to base model
    if run_alignment_year_one:
        print(f"ALIGNING {year_pair[0].stem}")

        first_year_model_paths = list(Path(f"{year_pair[0]}").rglob("*model"))
        year_models_one = list(
            map(lambda x: Word2Vec.load(str(x)), first_year_model_paths)
        )
        year_models_one = align_word2vec_models(base_model, year_models_one)
        temp_year_one_path.mkdir(exist_ok=True, parents=True)

        for model, label in zip(year_models_one, first_year_model_paths):
            model.wv.save(f"{temp_year_one_path}/{label.stem}_aligned.kv")

    if run_alignment_year_two:
        print(f"ALIGNING {year_pair[1].stem}")

        second_year_model_paths = list(Path(f"{year_pair[1]}").rglob("*model"))
        year_models_two = list(
            map(lambda x: Word2Vec.load(str(x)), second_year_model_paths)
        )
        year_models_two = align_word2vec_models(base_model, year_models_two)
        temp_year_two_path.mkdir(exist_ok=True, parents=True)

        for model, label in zip(year_models_two, second_year_model_paths):
            model.wv.save(f"{temp_year_two_path}/{label.stem}_aligned.kv")

    # Calculate the intra-year distances
    # Year One
    if run_intra_year_one:
        print(f"CALCULATING INTRA YEAR {year_pair[0].stem}")

        first_year_model_paths = list(temp_year_one_path.rglob("*kv"))
        year_models_one = list(
            map(lambda x: KeyedVectors.load(str(x)), first_year_model_paths)
        )
        batch_iterator = build_batch_filelist(
            year_models_one, first_year_model_paths, intra_year_one_path
        )

        with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
            processed_distances_files = pool.map(calculate_distances, batch_iterator)

        merge_calculated_distances(processed_distances_files, str(intra_year_one_path))

    # Year Two
    if run_intra_year_two:
        print(f"CALCULATING INTRA YEAR {year_pair[1].stem}")

        second_year_model_paths = list(temp_year_two_path.rglob("*kv"))
        year_models_two = list(
            map(lambda x: KeyedVectors.load(str(x)), second_year_model_paths)
        )
        batch_iterator = build_batch_filelist(
            year_models_two, second_year_model_paths, intra_year_two_path
        )

        with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
            processed_distances_files = pool.map(calculate_distances, batch_iterator)

        merge_calculated_distances(processed_distances_files, str(intra_year_two_path))

    # Calculate inter-year distances
    if run_inter_year:
        print(f"CALCULATING INTER YEAR {year_pair[0].stem}-{year_pair[1].stem}")

        # Load the files if intra-year not needed
        if not run_intra_year_one:
            first_year_model_paths = list(temp_year_one_path.rglob("*kv"))
            year_models_one = list(
                map(lambda x: KeyedVectors.load(str(x)), first_year_model_paths)
            )
        if not run_intra_year_two:

            second_year_model_paths = list(temp_year_two_path.rglob("*kv"))
            year_models_two = list(
                map(lambda x: KeyedVectors.load(str(x)), second_year_model_paths)
            )

        batch_iterator = build_batch_filelist(
            year_models_one,
            first_year_model_paths,
            inter_year_path,
            year_models_two,
            second_year_model_paths,
        )

        with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
            processed_distance_files = pool.map(calculate_distances, batch_iterator)

        merge_calculated_distances(processed_distance_files, str(inter_year_path))
