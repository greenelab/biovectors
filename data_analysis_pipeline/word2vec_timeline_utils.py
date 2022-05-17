import copy
import csv
import itertools
from pathlib import Path
import multiprocessing
import sys
from typing import Sequence, Any, Iterable

from gensim.models import Word2Vec
import gensim.models.word2vec as w2v
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cdist
import tqdm


class SentenceIterator:
    """
    This class is designed to take each outputted sentence and feed them into a word2vec model as input
    """

    def __init__(self, sentences: list, seed: int = 100):
        self.sentences = sentences
        self.seed = seed

        self.sentence_indicies = list(range(len(self.sentences)))
        np.random.seed(seed)

    def __iter__(self):
        np.random.shuffle(self.sentence_indicies)
        for sen_idx in tqdm.tqdm(
            self.sentence_indicies,
            desc=f"Model {self.seed-100}",
            position=self.seed - 100,
        ):
            yield self.sentences[sen_idx].split(" ")


def align_word2vec_models(
    base_model: Word2Vec, models_to_align: Sequence[Word2Vec]
) -> Word2Vec:
    """
    This function is designed to align word vectors onto the base word vector model.
    based on https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    """
    # Get normed vector first
    base_model.init_sims()

    aligned_model_arr = []
    for model_to_align in tqdm.tqdm(models_to_align):

        # Insure that the original object passed doesn't get changed
        # base_model = copy.deepcopy(base_model)
        model_to_align = copy.deepcopy(model_to_align)

        # Get shared vocabulary words
        base_model_vocab = set(base_model.wv.vocab.keys())
        model_to_align_vocab = set(model_to_align.wv.vocab.keys())

        # Sort based on frequency of both models
        common_vocab = base_model_vocab & model_to_align_vocab
        common_vocab = list(common_vocab)
        common_vocab.sort(
            key=lambda word: model_to_align.wv.vocab[word].count,
            reverse=True,
        )

        # Resort word vectors based on frequency and
        # Replace the vectors themselves
        model_to_align_indicies = [
            model_to_align.wv.vocab[word].index for word in common_vocab
        ]

        model_to_align.init_sims()
        old_arr = model_to_align.wv.vectors_norm
        new_arr = np.array([old_arr[index] for index in model_to_align_indicies])

        # Calculate orthogonal procrustes
        translation_matrix, _ = orthogonal_procrustes(
            new_arr, base_model.wv[common_vocab]
        )
        model_to_align.wv.vectors = new_arr @ translation_matrix

        # Update the wordvector object to reflect new changes
        new_vocab = {}
        for index, word in enumerate(common_vocab):
            new_vocab[word] = w2v.Vocab(
                index=index, count=model_to_align.wv.vocab[word].count
            )

        model_to_align.wv.vocab = new_vocab

        aligned_model_arr.append(model_to_align)

    # Return updated keyedvectors
    return aligned_model_arr


def build_batch_filelist(
    year_one_models,
    year_one_model_paths,
    output_filepath,
    year_two_models=None,
    year_two_model_paths=None,
):
    """
    This function build the batch iterator for calculating the inter and intra year distances.
    Args:
        year_one_models - the aligned word vector models
        year_one_model_paths - the paths to the wordvector models to keep track of which model pair is being used
        output_filepath - the outfile for the temp files to be written to
        year_two_models - the second set of word vector models (optional if calculating intra-distances)
        year_two_model_paths - the second set of word vector models paths (optional if calculating intra-distances)
    """
    year_two_model_passed = year_two_models is not None
    year_two_model_paths_passed = year_two_model_paths is not None

    if year_two_model_passed != year_two_model_paths_passed:
        raise Exception(
            "Please make sure to pass in both the models and their respective paths"
        )

    # Pythons way of checking to see if all variables are True
    elif all([year_two_model_passed, year_two_model_paths_passed]):
        return [
            (
                model_one[0],
                model_two[0],
                "_".join(model_one[1].stem.split("_")[0:2]),
                "_".join(model_two[1].stem.split("_")[0:2]),
                output_filepath.parent,
            )
            for model_one, model_two in itertools.product(
                zip(year_one_models, year_one_model_paths),
                zip(year_two_models, year_two_model_paths),
            )
        ]

    else:
        return [
            (
                model_one[0],
                model_two[0],
                "_".join(model_one[1].stem.split("_")[0:2]),
                "_".join(model_two[1].stem.split("_")[0:2]),
                output_filepath.parent,
            )
            for model_one, model_two in itertools.combinations(
                zip(year_one_models, year_one_model_paths), r=2
            )
        ]


def calculate_distances(argument_batch):
    """
    This function is designed to calculate distances for each model pair.
    It also writes the output to a file.

    Argument:
        argument_batch - a tuple that contains arguments for running this function (designed for parallel processing)
    """

    model_one = argument_batch[0]
    model_two = argument_batch[1]
    year_one = argument_batch[2]
    year_two = argument_batch[3]
    parent_folder = argument_batch[4]

    temp_filename = (
        f"{str(parent_folder)}/calculated_{year_one}-{year_two}_intra_distance_temp.tsv"
    )

    common_vocab = set(model_one.wv.vocab.keys()) & set(model_two.wv.vocab.keys())

    with open(temp_filename, "w") as outfile:
        writer = csv.DictWriter(
            outfile, delimiter="\t", fieldnames=["tok", "year_pair", "distance"]
        )
        writer.writeheader()
        for tok in tqdm.tqdm(common_vocab):
            writer.writerow(
                dict(
                    tok=tok,
                    year_pair=f"{year_one}-{year_two}",
                    distance=cdist([model_one.wv[tok]], [model_two.wv[tok]], "cosine")[
                        0
                    ][0],
                )
            )

    return temp_filename


def merge_calculated_distances(temp_filepaths, outfile_path):
    with open(outfile_path, "w") as outfile:
        writer = csv.DictWriter(
            outfile, delimiter="\t", fieldnames=["tok", "year_pair", "distance"]
        )
        writer.writeheader()
        for filename in temp_filepaths:
            with open(filename, "r") as infile:
                reader = csv.DictReader(
                    infile, delimiter="\t", fieldnames=["tok", "year_pair", "distance"]
                )
                for entry in tqdm.tqdm(reader):
                    writer.writerow(entry)
            Path(filename).unlink(missing_ok=True)


def train_word2vec(doc_iterator):
    sentence_iterator = doc_iterator[0]
    output_folder = doc_iterator[1]
    seed = doc_iterator[2]
    seed_idx = doc_iterator[3]

    seed = seed + seed_idx
    model = Word2Vec(size=300, seed=seed, window=16, workers=4, min_count=5)
    model.build_vocab(sentence_iterator)
    model.train(sentence_iterator, epochs=5, total_examples=model.corpus_count)
    model.save(
        f"{str(output_folder)}/{output_folder.stem.split('_')[0]}_{seed_idx}.model"
    )


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


def write_to_file(doc_iterator):

    with open(
        f"{doc_iterator[2]}/{doc_iterator[1]}_fulltext_output.txt", "w"
    ) as full_text_outfile:
        for parsed_text in doc_iterator[0]:
            full_text_outfile.write(parsed_text)
            full_text_outfile.write("\n")
