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

# # Pubtator Central Update

# Pubtator Central updates their data monthly; however, they changed their data to be xml format instead of common text.
# Based on this update it is imperative to know what changes have been made/format this data to make future experiments easier to work with.
# This notebook is being created to make training word2vec a whole lot easier.

# +
# %load_ext autoreload
# %autoreload 2

from collections import defaultdict, Counter
import csv
from datetime import datetime
import itertools
import lzma
from pathlib import Path
import pickle
import tarfile

import lxml.etree as ET
import numpy as np
import pandas as pd
import plotnine as p9
import tqdm

from biovectors_modules.word2vec_run_helper import (
    PubMedSentencesIterator,
    PubtatorTarIterator,
    chunks,
)
# -

# # Look at xml example

# Lets look at an example entry for each tagged document. Looks like it is regular BioCXML format which is good for processing.

# Move up a level in the repository, enter the folder with all the pubmed abstracts
# grab all files with the .gz extenstion for processing
pubtator_abstract_batch = list(Path("../pubtator_abstracts").rglob("*.gz"))
print(len(pubtator_abstract_batch))

for batch_directory in pubtator_abstract_batch:
    for doc_obj in PubtatorTarIterator(batch_directory):
        passages = doc_obj.xpath("//passage")
        lxml_str = ET.tostring(passages[1], pretty_print=True)
        print(lxml_str.decode("utf-8"))
        break
    break

# # Grab Document Metadata

if not Path("output/pmc_metadata.tsv.xz").exists():
    with lzma.open("output/pmc_metadata.tsv.xz", "wt") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "batch_folder",
                "doc_id",
                "doi",
                "pmc",
                "pmid",
                "section",
                "published_year",
            ],
            delimiter="\t",
        )
        writer.writeheader()

        # Cycle through each batch file
        for batch_directory in pubtator_abstract_batch:

            # Cycle through each document
            for doc_obj in tqdm.tqdm(PubtatorTarIterator(batch_directory)):
                doc_id = doc_obj.xpath("id/text()")
                doi = doc_obj.xpath("passage/infon[@key='article-id_doi']/text()")
                pmc_id = doc_obj.xpath("passage/infon[@key='article-id_pmc']/text()")
                pmid = doc_obj.xpath("passage/infon[@key='article-id_pmid']/text()")
                sections = doc_obj.xpath("passage/infon[@key='section_type']/text()")
                section_type = doc_obj.xpath("passage/infon[@key='type']/text()")
                year = doc_obj.xpath("passage/infon[@key='year']/text()")

                section_headers = "|".join(sorted(list(set(sections))))
                section_headers_type = "|".join(
                    sorted(list(set(map(lambda x: x.upper(), section_type))))
                )

                writer.writerow(
                    {
                        "batch_folder": batch_directory.name,
                        "doc_id": doc_id[0],
                        "doi": doi[0] if len(doi) > 0 else "",
                        "pmc": pmc_id[0] if len(pmc_id) > 0 else "",
                        "pmid": pmid[0] if len(pmid) > 0 else "",
                        "section": section_headers
                        if len(section_headers) > 0
                        else section_headers_type,
                        "published_year": year[0]
                        if len(year) > 0
                        else 0,  # Shouldnt get to the else statement
                    }
                )


# # Analyze Abstract/Full Text Dataset

pubtator_central_metadata_df = pd.read_csv("output/pmc_metadata.tsv.xz", sep="\t")
print(pubtator_central_metadata_df.shape)
pubtator_central_metadata_df.head()

# ## Sanity Check the data

# Sanity check that all documents have a published year
(pubtator_central_metadata_df.query("published_year.isnull()").shape)

# Do all documents have an id?
(pubtator_central_metadata_df.query("doc_id.isnull()").shape)

# Do all documents have a pmid?
(pubtator_central_metadata_df.query("pmid.isnull()").shape)

# Do all documents have a pmc id?
(pubtator_central_metadata_df.query("pmc.isnull()").shape)

# Do all documents have a doi?
(pubtator_central_metadata_df.query("doi.isnull()").shape)

# ## Published Year Distribution

(pubtator_central_metadata_df.sort_values("published_year").published_year.unique())

doc_count_df = (
    pubtator_central_metadata_df.groupby("published_year")
    .agg({"published_year": "size"})
    .rename(index=str, columns={"published_year": "doc_count"})
    .reset_index()
    .astype({"published_year": int, "doc_count": int})
)
doc_count_df.head()

g = (
    p9.ggplot(
        doc_count_df.query("published_year > 0& published_year < 1950"),
        p9.aes(x="published_year", y="doc_count"),
    )
    + p9.geom_col(position=p9.position_dodge(width=0.9), fill="#1f78b4")
    + p9.labs(
        title="Number of Documents Pre 1950", x="Publication Year", y="Document Count"
    )
)
g.save("output/figures/pre_1950_doc_count.png", dpi=500)
print(g)

g = (
    p9.ggplot(
        doc_count_df.query("published_year >= 1950"),
        p9.aes(x="published_year", y="doc_count"),
    )
    + p9.geom_col(position=p9.position_dodge(width=0.9), fill="#1f78b4")
    + p9.labs(
        title="Number of Documents Post 1950", x="Publication Year", y="Document Count"
    )
)
g.save("output/figures/post_1950_doc_count.png", dpi=500)
print(g)

# # Shared Tokens Across Time - Abstract Only

tokens_by_year = defaultdict(Counter)
sentence_iterator = PubMedSentencesIterator(
    pubtator_abstract_batch,
    year_filter=list(range(1990, datetime.now().year + 1, 1)),
    return_year=True,
    jobs=3,
)

if not Path("output/unique_tokens_by_year.pkl").exists():
    for year, sentence in tqdm.tqdm(sentence_iterator):
        tokens_by_year[year].update(Counter(sentence))

if not Path("output/unique_tokens_by_year.pkl").exists():
    pickle.dump(tokens_by_year, open("output/unique_tokens_by_year.pkl", "wb"))
else:
    tokens_by_year = pickle.load(open("output/unique_tokens_by_year.pkl", "rb"))

# ## Unique Tokens Available per Year

# +
data_rows = []

for query_year in tokens_by_year:

    data_rows.append(
        {
            "year": query_year,
            "num_tokens": len(tokens_by_year[query_year]),
        }
    )
# -

unique_token_df = pd.DataFrame.from_records(data_rows)
unique_token_df

g = (
    p9.ggplot(unique_token_df, p9.aes(x="year", y="num_tokens"))
    + p9.geom_col(fill="#1f78b4")
    + p9.coord_flip()
    + p9.labs(
        title="Number of Abstract Tokens Available Post 1990",
        x="Year",
        y="# Unique Tokens",
    )
)
g.save("output/figures/post_1990_unique_tokens_abstracts.png", dpi=500)
print(g)

# ## Shared tokens across years

# +
data_rows = []
reversed_tokens = list(sorted(tokens_by_year.keys()))[::-1]
all_tokens = set(tokens_by_year[2021].keys()) | set(tokens_by_year[2020].keys())

for query_year in reversed_tokens[1:]:
    query_year_vocab_set = set(tokens_by_year[query_year].keys())
    tokens_matched = all_tokens & query_year_vocab_set

    data_rows.append(
        {
            "years": str(query_year) if query_year != 2020 else "2020-21",
            "percentage_tokens_mapped": len(tokens_matched) / len(all_tokens),
            "num_tokens_matched": len(tokens_matched),
            "num_tokens_total": len(all_tokens),
        }
    )
# -

token_overlap_df = pd.DataFrame.from_dict(data_rows)
token_overlap_df

g = (
    p9.ggplot(
        token_overlap_df.iloc[1:, :], p9.aes(x="years", y="percentage_tokens_mapped")
    )
    + p9.geom_col(fill="#1f78b4")
    + p9.coord_flip()
    + p9.labs(
        title="Token Overlap with 2020-2021 Abstracts",
        x="Year",
        y="% Tokens Overlapped",
    )
)
g.save("output/figures/tokens_overlap_with_2020-21_abstracts.png", dpi=500)
print(g)

# # Shared Tokens Across Time - Full Text Only

# Grab sentences within the full text documents

tokens_by_year_full_text = defaultdict(Counter)
sentence_iterator = PubMedSentencesIterator(
    pubtator_abstract_batch,
    section_filter=["INTRO", "METHODS", "RESULTS", "DISCUSS", "CONCL", "SUPPL"],
    year_filter=list(range(1990, datetime.now().year + 1, 1)),
    return_year=True,
    jobs=3,
)

if not Path("output/unique_tokens_by_year_full_text.pkl").exists():
    for year, sentence in tqdm.tqdm(sentence_iterator):
        tokens_by_year_full_text[year].update(Counter(sentence))

if not Path("output/unique_tokens_by_year_full_text.pkl").exists():
    pickle.dump(
        tokens_by_year_full_text,
        open("output/unique_tokens_by_year_full_text.pkl", "wb"),
    )
else:
    tokens_by_year_full_text = pickle.load(
        open("output/unique_tokens_by_year_full_text.pkl", "rb")
    )

# ## Unique Tokens Available per Year

# +
data_rows = []

for query_year in tokens_by_year_full_text:

    data_rows.append(
        {
            "year": query_year,
            "num_tokens": len(tokens_by_year_full_text[query_year]),
        }
    )
# -

unique_token_full_text_df = pd.DataFrame.from_records(data_rows)
unique_token_full_text_df

g = (
    p9.ggplot(unique_token_full_text_df, p9.aes(x="year", y="num_tokens"))
    + p9.geom_col(fill="#1f78b4")
    + p9.coord_flip()
    + p9.labs(
        title="Number of Full Text Tokens Available Post 1990",
        x="Year",
        y="# Unique Tokens",
    )
)
g.save("output/figures/post_1990_unique_tokens_full_text.png", dpi=500)
print(g)

# ## Shared tokens across years

# +
data_rows = []
reversed_tokens = list(sorted(tokens_by_year_full_text.keys()))[::-1]
all_tokens = set(tokens_by_year_full_text[2021].keys()) | set(
    tokens_by_year_full_text[2020].keys()
)

for query_year in reversed_tokens[1:]:
    query_year_vocab_set = set(tokens_by_year_full_text[query_year].keys())
    tokens_matched = all_tokens & query_year_vocab_set

    data_rows.append(
        {
            "years": str(query_year) if query_year != 2020 else "2020-21",
            "percentage_tokens_mapped": len(tokens_matched) / len(all_tokens),
            "num_tokens_matched": len(tokens_matched),
            "num_tokens_total": len(all_tokens),
        }
    )
# -

token_overlap_full_text_df = pd.DataFrame.from_dict(data_rows)
token_overlap_full_text_df

g = (
    p9.ggplot(
        token_overlap_full_text_df.iloc[1:, :],
        p9.aes(x="years", y="percentage_tokens_mapped"),
    )
    + p9.geom_col(fill="#1f78b4")
    + p9.coord_flip()
    + p9.labs(
        title="Token Overlap with 2020-2021 Full Text",
        x="Year",
        y="% Tokens Overlapped",
    )
)
g.save("output/figures/tokens_overlap_with_2020-21_full_text.png", dpi=500)
print(g)
