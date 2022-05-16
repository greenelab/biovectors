import argparse
from pathlib import Path
import multiprocessing
import re
import tarfile

from gensim.models import Word2Vec
from tar_iterators import AbstractIterator

from word2vec_timeline_utils import write_to_file


parser = argparse.ArgumentParser(description="Output annotated BioC xml to plain text.")
parser.add_argument(
    "--input_folder", help="The folder that contains xml files grouped by year"
)
parser.add_argument(
    "--output_folder", help="The folder that contains xml files grouped by year"
)
parser.add_argument(
    "--skip_section_check",
    action="store_true",
    help="Tell the program to skip checking for sections",
)

argv = parser.parse_args()

input_folder = argv.input_folder
output_folder = argv.output_folder

abstract_only_sections = {
    "TITLE",
    "TITLE_1",
    "TITLE_2",
    "TITLE_3",
    "TITLE_4",
    "TITLE_5",
    "TITLE_CAPTION",
    "ABSTRACT",
}

full_text_sections = {
    "TITLE",
    "TITLE_1",
    "TITLE_2",
    "TITLE_3",
    "TITLE_4",
    "TITLE_5",
    "TITLE_CAPTION",
    "ABSTRACT",
    "INTRO",
    "METHODS",
    "RESULTS",
    "DISCUSS",
    "CONCL",
    "SUPPL",
    "CASE",
    "PARAGRAPH",
}

year_folders = Path(input_folder).glob("*")

print("Time to parse the documents")
years_to_iterate = sorted(list(year_folders))
print(years_to_iterate)

# Filter years already parsed
years_to_iterate = [
    year
    for year in years_to_iterate
    if not Path(f"{output_folder}/{year.stem}_fulltext_output.txt").exists()
]
print(len(years_to_iterate))
if len(years_to_iterate) <= 1:
    doc_iterators = [
        (
            AbstractIterator(
                list(Path(f"{input_folder}/{year.stem}").rglob("*xml")),
                section_filter=full_text_sections,
                skip_section_check=argv.skip_section_check,
            ),
            year.stem,
            output_folder,
        )
        for year in years_to_iterate
    ]
    write_to_file(doc_iterators[0])

else:
    with multiprocessing.Pool(processes=3) as pool:
        doc_iterators = [
            (
                AbstractIterator(
                    list(Path(f"{input_folder}/{year.stem}").rglob("*xml")),
                    section_filter=full_text_sections,
                    skip_section_check=argv.skip_section_check,
                ),
                year.stem,
                output_folder,
            )
            for year in years_to_iterate
        ]
        pool.map(write_to_file, doc_iterators)
