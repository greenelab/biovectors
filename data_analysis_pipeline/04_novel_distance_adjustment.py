import argparse
import csv
from pathlib import Path
import sys

from gensim.models import Word2Vec
import tqdm

parser = argparse.ArgumentParser(
    description="Construct the novel distance metric for changepoint detection."
)
parser.add_argument(
    "--intra_year_folder", help="The folder that contains the intra-year calculations."
)
parser.add_argument(
    "--inter_year_folder", help="The folder that contains inter-year calculations."
)
parser.add_argument(
    "--output_folder", help="The folder to output the novel distance calculations."
)
parser.add_argument(
    "--trained_model_folder",
    help="the folder that contains the trained word2vec models.",
)

argv = parser.parse_args()

inter_year_calculations = list(Path(argv.inter_year_folder).rglob("*tsv"))
intra_year_calculations = {
    model.stem.split("_")[1]: model
    for model in Path(argv.intra_year_folder).rglob("*tsv")
}

final_folder = Path(argv.output_folder)
final_folder.mkdir(exist_ok=True, parents=True)

for inter_year_pair in inter_year_calculations:
    token_adjusted_distances = dict()
    year_pair = inter_year_pair.stem.split("_")[1].split(
        "-"
    )  # split for e.g. 2019-2020

    output_file = Path(
        f"{final_folder}/final_{year_pair[0]}-{year_pair[1]}_distance.tsv"
    )

    if output_file.exists():
        continue

    with open(inter_year_pair, "r") as inter_infile:
        inter_year_reader = csv.DictReader(inter_infile, delimiter="\t")

        for line in tqdm.tqdm(inter_year_reader):
            token = line["tok"]

            if line["distance"] == "distance":
                continue

            if token not in token_adjusted_distances:
                token_adjusted_distances[token] = dict()
                token_adjusted_distances[token]["inter_year_distance"] = 0
                token_adjusted_distances[token]["inter_count"] = 0

            token_adjusted_distances[token]["inter_year_distance"] += float(
                line["distance"]
            )
            token_adjusted_distances[token]["inter_count"] += 1

    with open(intra_year_calculations[year_pair[0]], "r") as intra_year_one_infile:
        intra_year_one_reader = csv.DictReader(intra_year_one_infile, delimiter="\t")
        for line in tqdm.tqdm(intra_year_one_reader):
            token = line["tok"]

            if token not in token_adjusted_distances:
                continue

            if line["distance"] == "distance":
                continue

            if "intra_year_one_distance" not in token_adjusted_distances[line["tok"]]:
                token_adjusted_distances[token]["intra_year_one_distance"] = 0
                token_adjusted_distances[token]["intra_one_count"] = 0

            token_adjusted_distances[token]["intra_year_one_distance"] += float(
                line["distance"]
            )
            token_adjusted_distances[token]["intra_one_count"] += 1

    with open(intra_year_calculations[year_pair[1]], "r") as intra_year_two_infile:
        intra_year_two_reader = csv.DictReader(intra_year_two_infile, delimiter="\t")
        for line in tqdm.tqdm(intra_year_two_reader):
            token = line["tok"]

            if token not in token_adjusted_distances:
                continue

            if line["distance"] == "distance":
                continue

            if "intra_year_two_distance" not in token_adjusted_distances[line["tok"]]:
                token_adjusted_distances[token]["intra_year_two_distance"] = 0
                token_adjusted_distances[token]["intra_two_count"] = 0

            token_adjusted_distances[token]["intra_year_two_distance"] += float(
                line["distance"]
            )
            token_adjusted_distances[token]["intra_two_count"] += 1

    trained_model_folder = argv.trained_model_folder
    with open(output_file, "w") as outfile:
        final_writer = csv.DictWriter(
            outfile,
            delimiter="\t",
            fieldnames=[
                "tok",
                "year_pair",
                "adjusted_distance",
                "frequency_ratio",
                "averaged_inter_distance",
                f"averaged_{year_pair[0]}_distance",
                f"averaged_{year_pair[1]}_distance",
            ],
        )
        final_writer.writeheader()

        model_one = Word2Vec.load(
            str(
                list(
                    Path(f"{trained_model_folder}/{year_pair[0]}_model").rglob(
                        "*0_*.model"
                    )
                )[0]
            )
        )
        model_two = Word2Vec.load(
            str(
                list(
                    Path(f"{trained_model_folder}/{year_pair[1]}_model").rglob(
                        "*0_*.model"
                    )
                )[0]
            )
        )

        for tok in token_adjusted_distances:
            final_inter_dist = (
                token_adjusted_distances[tok]["inter_year_distance"]
                / token_adjusted_distances[tok]["inter_count"]
            )
            final_intra_one_dist = (
                token_adjusted_distances[tok]["intra_year_one_distance"]
                / token_adjusted_distances[tok]["intra_one_count"]
            )
            final_intra_two_dist = (
                token_adjusted_distances[tok]["intra_year_two_distance"]
                / token_adjusted_distances[tok]["intra_two_count"]
            )

            final_writer.writerow(
                {
                    "tok": tok,
                    "year_pair": "-".join(year_pair),
                    "adjusted_distance": final_inter_dist
                    / (final_intra_one_dist + final_intra_two_dist),
                    "frequency_ratio": float(model_two.wv.vocab[tok].count)
                    / float(model_one.wv.vocab[tok].count),
                    "averaged_inter_distance": final_inter_dist,
                    f"averaged_{year_pair[0]}_distance": final_intra_one_dist,
                    f"averaged_{year_pair[1]}_distance": final_intra_two_dist,
                }
            )
