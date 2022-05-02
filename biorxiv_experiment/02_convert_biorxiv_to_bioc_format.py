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

# # Convert biorxiv and medrxiv into Pubtator's bioC format

# Convert biorxiv and medrxiv preprints into Pubtator's BioCXML format, so Pubtator's taggers can work.
# This notebook only uses the most current version of a preprint.

# +
from pathlib import Path

import lxml.etree as ET
import pandas as pd
import plydata as ply
import plydata.tidy as ply_tdy
import tqdm

from biovectors_modules.preprint_converter import convert_to_bioc
# -

biorxiv_folder = Path("output/biorxiv_medrxiv_dump/biorxiv")
medrxiv_folder = Path("output/biorxiv_medrxiv_dump/medrxiv")

# ## Get the time stamps of all medrxiv and biorxiv preprints

if not Path("output/all_medrxiv_biorxiv_timestamps.tsv").exists():
    data_rows = []
    xml_parser = ET.XMLParser(encoding="UTF-8", recover=True)
    all_biomed_preprints = list(biorxiv_folder.rglob("*xml")) + list(
        medrxiv_folder.rglob("*xml")
    )
    for xml_file in tqdm.tqdm(all_biomed_preprints):
        doc_obj = ET.parse(str(xml_file), parser=xml_parser).getroot()
        date_row = doc_obj.xpath("//date[@date-type='accepted']/*/text()")
        attribute_label = "accepted"
        if len(date_row) == 0:
            date_row = doc_obj.xpath("//date[@date-type='received']/*/text()")
            attribute_label = "received"
        # In case there is no date found
        if len(date_row) == 0:
            doc_date = "NA"
        else:
            doc_date = "-".join([date_row[1], date_row[0], date_row[2]])
        data_rows.append(
            {
                "date": doc_date,
                "repository": xml_file.parents[0].name,
                "doc_id": xml_file.stem,
                "attribute": attribute_label,
            }
        )

if not Path("output/all_medrxiv_biorxiv_timestamps.tsv").exists():
    preprint_dates_df = (
        pd.DataFrame.from_records(data_rows)
        >> ply_tdy.separate("date", into=["month", "day", "year"])
        >> ply_tdy.separate("doc_id", into=["doc_id", "version"])
    )
    preprint_dates_df >> ply.call(
        ".to_csv", "output/all_medrxiv_biorxiv_timestamps.tsv", sep="\t", index=False
    )
else:
    preprint_dates_df = pd.read_csv(
        "output/all_medrxiv_biorxiv_timestamps.tsv", sep="\t", keep_default_na=False
    )
preprint_dates_df >> ply.slice_rows(10)

# ## Filter each preprint to the latest version

preprint_dates_df.sort_values("year").year.value_counts()

latest_version_df = (
    (
        preprint_dates_df
        >> ply.arrange("doc_id", "version")
        >> ply.call(".groupby", "doc_id")
    ).agg(
        {
            "version": "last",
            "month": "last",
            "day": "last",
            "year": "last",
            "repository": "last",
        }
    )
    >> ply.call(".reset_index")
    >> ply.define(
        doc_id=lambda x: x.doc_id.apply(
            lambda y: "%06d" % (int(y)) if type(y) == int else y
        )
    )
    >> ply_tdy.unite("doc_id_version", "doc_id", "version", sep="_")
    >> ply.rename(doc_id="doc_id_version")
)
latest_version_df >> ply.slice_rows(10)

# ## Perform the actual conversion to BioCXML format

filter_tag_list = [
    "sc",
    "italic",
    "sub",
    "inline-formula",
    "disp-formula",
    "bold",
    "tr",
    "td",
]

for doc, year, repository in tqdm.tqdm(
    latest_version_df >> ply.pull(["doc_id", "year", "repository"])
):

    doc_folder = medrxiv_folder if repository == "medrxiv" else biorxiv_folder

    parser = ET.XMLParser(encoding="UTF-8", recover=True, remove_blank_text=True)
    tree = ET.parse(open(f"{doc_folder}/{doc}.xml", "rb"), parser=parser)
    ET.strip_tags(tree, *filter_tag_list)

    converted_tree = convert_to_bioc(tree, repository=repository)
    output_folder = Path(f"output/converted_docs/{year}")
    output_folder.mkdir(exist_ok=True, parents=True)
    ET.ElementTree(converted_tree).write(
        f"{output_folder}/{doc}.{doc_folder.stem}.bioc.xml",
        pretty_print=True,
        method="c14n",
    )
