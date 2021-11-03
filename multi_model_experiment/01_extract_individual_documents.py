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

# # Sort Documents into Individual Folders

# +
from pathlib import Path
import tarfile

import lxml.etree as ET
import pandas as pd
import tqdm
# -

# # Grab the TarFiles in Directory

tarfiles = list(Path("../pubtator_abstracts").rglob("*gz"))
tarfiles[0:2]

# # Load the document Metadata

pmc_metadata_df = pd.read_csv(
    "../exploratory_data_analysis/output/pmc_metadata.tsv.xz", sep="\t"
)
pmc_metadata_df.head()

for tar_obj in tarfiles:
    open_tar = tarfile.open(str(tar_obj))

    while True:
        try:
            batch_file = open_tar.next()

            if batch_file is None:
                break

            doc_generator = ET.iterparse(
                open_tar.extractfile(batch_file), tag="document", recover=True
            )

            for event, doc_obj in tqdm.tqdm(doc_generator):
                year = doc_obj.xpath(
                    "passage[contains(infon[@key='section_type']/text(), 'TITLE')]/infon[@key='year']/text()"
                )

                if len(year) < 1 or int(year[0]) < 2000:
                    continue

                year_folder = Path("output") / Path(year[0])
                if not year_folder.exists():
                    year_folder.mkdir()

                pmid = doc_obj.xpath("passage/infon[@key='article-id_pmid']/text()")

                with open(f"output/abstract_output/{year[0]}/{pmid[0]}.xml", "wb") as f:
                    f.write(ET.tostring(doc_obj, pretty_print=True, encoding="utf-8"))

        except tarfile.ReadError:
            print(f"broke out of {str(tar_obj)}")
            break
