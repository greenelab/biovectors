# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1+dev
#   kernelspec:
#     display_name: Python [conda env:biovectors]
#     language: python
#     name: conda-env-biovectors-py
# ---

# # Train Multiple Word2vec Models at One Time

# +
import multiprocessing
from pathlib import Path
import random
import re
import time

from gensim.models import Word2Vec
import lxml.etree as ET
import spacy


# -

# # Set up Document Iterator

class AbstractIterator:
    def __init__(self, file_iterator, section_filter=None, seed=100):
        self.file_iterator = file_iterator

        if section_filter is None:
            section_filter = ["TITLE", "ABSTRACT"]
        self.section_filter = section_filter

        disabled_pipelines = [
            "tagger",
            "parser",
            "ner",
            "attribute_ruler",
            "tok2vec",
        ]
        self.nlp = spacy.load("en_core_web_sm", disable=disabled_pipelines)
        random.seed(seed)

    def __iter__(self):
        random.shuffle(self.file_iterator)
        for doc in self.file_iterator:
            doc_obj = ET.parse(str(doc)).getroot()

            for passage in doc_obj.xpath("passage"):
                section = passage.xpath("infon[@key='section_type']/text()")

                if section[0] not in self.section_filter:
                    continue

                passage_text = passage.xpath("text/text()")

                if len(passage_text) < 1:
                    continue

                passage_text = passage_text[0]

                passage_offset = passage.xpath("offset/text()")[0]
                current_pos = 0
                yield_text = ""

                sorted_passages = sorted(
                    passage.xpath("annotation"),
                    key=lambda x: int(x.xpath("location")[0].attrib["offset"]),
                )

                for annotation in sorted_passages:
                    annot_identifier = annotation.xpath(
                        "infon[@key='identifier']/text()"
                    )

                    if len(annot_identifier) == 0 or annot_identifier[0] == "-":
                        continue

                    annot_text = annotation.xpath("text/text()")[0]
                    location = annotation.xpath("location")

                    # replace string with identifier
                    entity_start = int(location[0].attrib["offset"]) - int(
                        passage_offset
                    )
                    entity_end = entity_start + int(location[0].attrib["length"])
                    replacement_str = re.sub(r"\s+", "_", annot_text)
                    replacement_str = f" {replacement_str} "
                    yield_text += (
                        passage_text[current_pos:entity_start] + replacement_str
                    )
                    current_pos = entity_end

                yield_text += passage_text[current_pos:]
                analyzed_text = self.nlp(yield_text.lower())

                yield list(map(str, analyzed_text))

            doc_obj.clear()


def train_word2vec(doc_iterator):
    seed = 100 + doc_iterator[2]
    model = Word2Vec(size=300, seed=seed, window=16, workers=5, min_count=1)
    model.build_vocab(doc_iterator[0])
    model.train(doc_iterator[0], epochs=10, total_examples=model.corpus_count)
    Path(f"output/models/{doc_iterator[1]}").mkdir(parents=True, exist_ok=True)
    model.save(
        f"output/models/{doc_iterator[1]}/{doc_iterator[1]}_{doc_iterator[2]}.model"
    )
    return f"finished {doc_iterator[2]}"


# # Train the Word2Vec models

folder_years = sorted(
    list(Path("output/abstract_output").iterdir()), key=lambda x: str(x).split("/")[1]
)

# create a pool of word2vec models
# Each pool trains the model and then see if you can train simultaenously
for year in folder_years:
    print(year)
    year_test = str(year).split("/")[1]
    if not Path(f"output/models/{year_test}").exists():
        num_of_models = 10
        with multiprocessing.Pool(processes=3) as pool:
            batch_files = list(year.rglob("*xml"))
            doc_iterators = [
                (AbstractIterator(batch_files), str(year).split("/")[1], idx)
                for idx in range(num_of_models)
            ]
            pool.map(train_word2vec, doc_iterators)
