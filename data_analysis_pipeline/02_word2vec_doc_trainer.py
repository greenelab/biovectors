import argparse
from pathlib import Path
import multiprocessing
import re

from gensim.models import Word2Vec
import numpy as np
import tqdm

from word2vec_timeline_utils import train_word2vec, SentenceIterator


parser = argparse.ArgumentParser(description="Train multiple Word2Vec models")
parser.add_argument(
    "--parse_file_path", help="The file that contains every sentence per line,"
)
parser.add_argument(
    "--model_cat",
    help="The category to label each model output. Typically it's (fulltext or abstract)",
)
parser.add_argument("--output_dir", help="The directory to output each trained model")
parser.add_argument(
    "--num_of_processes",
    help="the number of processes to train the word2vec models",
    nargs="?",
    const=1,
    type=int,
)

args = parser.parse_args()

num_of_models = 10
file_to_parse = Path(args.parse_file_path)
print(file_to_parse)

model_cat = args.model_cat

folder_name = file_to_parse.stem.split("_")[0]
output_folder = Path(f"{args.output_dir}/{folder_name}_model")
output_folder.mkdir(exist_ok=True, parents=True)

# Populate memory with enough processes to run multiple models
with multiprocessing.Pool(processes=args.num_of_processes) as pool:
    with open(file_to_parse, "r") as infile:
        sen_array = []
        for line in tqdm.tqdm(infile):
            if line.count(" ") < 2:
                continue

            sen_array.append(re.sub("\n", "", line.strip()))

    doc_iterators = [
        (
            SentenceIterator(sen_array, seed=100 + idx),
            output_folder,
            idx,
            model_cat,
        )
        for idx in range(num_of_models)
    ]
    pool.map(train_word2vec, doc_iterators)
