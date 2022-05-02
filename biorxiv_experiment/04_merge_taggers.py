# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # Merge GnormPlus and Tagger One Tags into One

# This tagging system used a divide and conquer approach, where each tagger tagged every preprint for their own respective entities.
# This notebook is designed to merge them back into one single file.
# Basically it aligns the passages and checks to see which entity comes first if present and then merges them into one single passage block.
# Once this process is finished preprints are ready to be processed by the main data analysis pipeline.

# +
from pathlib import Path

import lxml.etree as ET
from lxml.etree import XMLSyntaxError
import tqdm
# -

output_folder = Path("output/final")
gnormplus_folder = Path("output/gnormplus_tags")
tagger_one_folder = Path("output/tagger_one_tags/")

for gnormplus_folder_year in gnormplus_folder.glob("*"):

    for gnormplus_doc_path in tqdm.tqdm(
        gnormplus_folder_year.rglob("*xml"), desc=f"{gnormplus_folder_year.stem}"
    ):
        final_doc_name = (
            f"{output_folder}/{gnormplus_folder_year.stem}/{gnormplus_doc_path.name}"
        )

        if Path(final_doc_name).exists():
            continue

        gnormplus_doc = ET.parse(str(gnormplus_doc_path)).getroot()
        try:
            tagger_one_doc = ET.parse(
                f"{tagger_one_folder}/{gnormplus_folder_year.stem}/{gnormplus_doc_path.name}"
            ).getroot()
        except XMLSyntaxError:
            tagger_one_doc = ET.Element("document")

        merged_doc = ET.Element("document")
        merged_doc.append(gnormplus_doc.xpath("document/id")[0])  # Add the doc id

        tagger_one_passages = tagger_one_doc.xpath("document/passage")
        gnormplus_passages = gnormplus_doc.xpath("document/passage")

        running_annotation_id = 0

        if len(tagger_one_passages) == 0:
            for gnormplus_passage in gnormplus_passages:
                total_annotations = gnormplus_passage.xpath("annotation")

                # if no annotations add the passage and continue
                if len(total_annotations) == 0:
                    merged_doc.append(gnormplus_passage)
                    continue

                sorted(
                    total_annotations,
                    key=lambda x: int(x.xpath("location")[0].attrib["offset"]),
                )

                new_passage = ET.Element("passage")
                new_passage.extend(
                    [gnormplus_passage[0], gnormplus_passage[1], gnormplus_passage[2]]
                )
                for annotation in total_annotations:
                    annotation.attrib["id"] = str(running_annotation_id)
                    new_passage.append(annotation)
                    running_annotation_id += 1

                merged_doc.append(new_passage)

        elif len(gnormplus_passages) == 0:
            for tagger_one_passage in tagger_one_passages:
                total_annotations = tagger_one_passage.xpath("annotation")

                # if no annotations add the passage and continue
                if len(total_annotations) == 0:
                    merged_doc.append(tagger_one_passage)
                    continue

                sorted(
                    total_annotations,
                    key=lambda x: int(x.xpath("location")[0].attrib["offset"]),
                )

                new_passage = ET.Element("passage")
                new_passage.extend(
                    [
                        tagger_one_passage[0],
                        tagger_one_passage[1],
                        tagger_one_passage[2],
                    ]
                )
                for annotation in total_annotations:
                    annotation.attrib["id"] = str(running_annotation_id)
                    new_passage.append(annotation)
                    running_annotation_id += 1

                merged_doc.append(new_passage)
        else:
            passage_merge_generator = zip(tagger_one_passages, gnormplus_passages)
            for tagger_one_passage, gnormplus_passage in passage_merge_generator:
                total_annotations = tagger_one_passage.xpath(
                    "annotation"
                ) + gnormplus_passage.xpath("annotation")

                # if no annotations add the passage and continue
                if len(total_annotations) == 0:
                    merged_doc.append(gnormplus_passage)
                    continue

                sorted(
                    total_annotations,
                    key=lambda x: int(x.xpath("location")[0].attrib["offset"]),
                )

                new_passage = ET.Element("passage")
                new_passage.extend(
                    [gnormplus_passage[0], gnormplus_passage[1], gnormplus_passage[2]]
                )
                for annotation in total_annotations:
                    annotation.attrib["id"] = str(running_annotation_id)
                    annotation[0].attrib["key"] = "identifier"
                    new_passage.append(annotation)
                    running_annotation_id += 1

                merged_doc.append(new_passage)

        doc_header = ET.Element("collection")
        doc_header.extend([gnormplus_doc[0], gnormplus_doc[1], gnormplus_doc[2]])
        doc_header.append(merged_doc)
        Path(f"{output_folder}/{gnormplus_folder_year.stem}").mkdir(
            exist_ok=True, parents=True
        )
        ET.ElementTree(doc_header).write(
            final_doc_name, pretty_print=True, method="c14n"
        )
