import copy
import random
import re
from threading import Thread

import lxml.etree as ET
import numpy as np
import spacy
import tqdm


class AbstractIterator:
    def __init__(
        self,
        file_iterator,
        section_filter=None,
        seed=100,
        idx=0,
        skip_section_check=False,
    ):
        random.seed(seed)
        self.file_iterator = file_iterator
        self.idx = idx

        if section_filter is None:
            section_filter = ["TITLE", "ABSTRACT"]

        self.section_filter = section_filter

        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        self.skip_section_check = skip_section_check

    def __iter__(self):
        for doc in tqdm.tqdm(
            self.file_iterator, position=self.idx % 3, desc=f"Model {self.idx}"
        ):
            doc_obj = ET.parse(str(doc)).getroot()
            passage_text_tracker = []

            for passage in doc_obj.xpath(
                "passage|document/passage"
            ):  # account for preprint bioc conversion

                if not self.skip_section_check:
                    # see if the document uses the original tag attribute
                    section = passage.xpath("infon[@key='section_type']/text()")

                    # Otherwise use the new formatting attribute
                    if len(section) == 0:
                        section = passage.xpath("infon[@key='type']/text()")

                    # Some how there is an edge case where section still can't be found
                    if len(section) == 0:
                        continue

                    if section[0].upper() not in self.section_filter:
                        continue

                passage_text = passage.xpath("text/text()")

                if len(passage_text) < 1:
                    continue

                yield_text = ""
                passage_text = passage_text[0]

                passage_offset = passage.xpath("offset/text()")[0]
                current_pos = 0

                sorted_passages = list(
                    filter(
                        lambda x: len(x.xpath("location")) > 0,
                        passage.xpath("annotation"),
                    )
                )

                sorted_passages = sorted(
                    sorted_passages,
                    key=lambda x: int(x.xpath("location")[0].attrib["offset"]),
                )

                for annotation in sorted_passages:
                    annot_identifier = annotation.xpath(
                        "infon[translate(@key, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='identifier']/text()"
                    )

                    if len(annot_identifier) == 0 or annot_identifier[0] == "-":
                        continue

                    annot_type = annotation.xpath("infon[@key='type']/text()")

                    if len(annot_type) == 0 or annot_type[0] == "-":
                        continue

                    location = annotation.xpath("location")

                    if "length" not in location[0].attrib:
                        continue

                    # replace string with identifier
                    entity_start = int(location[0].attrib["offset"]) - int(
                        passage_offset
                    )

                    entity_end = entity_start + int(location[0].attrib["length"])
                    replacement_str = f" {annot_type[0].upper()}_{annot_identifier[0].replace(':','_')} "
                    yield_text += (
                        passage_text[current_pos:entity_start].lower() + replacement_str
                    )
                    current_pos = entity_end

                yield_text += passage_text[current_pos:]
                passage_text_tracker.append(yield_text)

            analyzed_text = self.nlp.pipe(passage_text_tracker)
            for passage in analyzed_text:
                for span in passage.sents:
                    yield " ".join(
                        [tok.lemma_.lower() for tok in span if tok.pos_ != "PUNCT"]
                    )

            doc_obj.clear()
