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

# # Run Word2vec on abstracts for each Decade

# +
from pathlib import Path

from gensim.models import Word2Vec, KeyedVectors
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import re
import tqdm

from biovectors_modules.word2vec_run_helper import (
    get_gene_disease_pairs,
    SentencesIterator,
    similarity_score,
)


# -

def get_year(pub_date):
    """
    Extracts four-digit year from publication's date string.
    """
    return re.search(r"\d\d\d\d", pub_date).group()


# ## Set up the Data

pairs = get_gene_disease_pairs(
    Path("output/hetnet_gene_disease_pairs.tsv"), Path("output/DO-slim-to-mesh.tsv")
)
pairs[0:10]

dates_df = pd.read_csv(
    Path("../pubmed_timestamp_experiment/output") / Path("pmid_to_pub_date.tsv.xz"),
    compression="xz",
    sep="\t",
)
print(dates_df.shape)
dates_df.head()

dates_df = (
    dates_df.dropna(subset=["pub_date"])
    .assign(pub_date=lambda x: x.pub_date.apply(get_year).astype(int))
    .query("~pub_date.isnull()")
)
print(dates_df.shape)
dates_df.head()

# ## Run the Models

# iterate through abstracts from 1971-2020 by decade
years = [1971, 1981, 1991, 2001, 2011]
Path("output/decades").mkdir(exists_ok=True)
Path("output/decades/models").mkdir(exists_ok=True)

for year in years:
    print(f"----- {str(year)} - {str(year+9)} -----")

    pmids = set(
        dates_df.loc[dates_df["pub_date"].between(year, year + 9), "pmid"].tolist()
    )

    if len(pmids) > 0:
        print(f"{len(pmids)} PMIDs from this time period")
        sentences = SentencesIterator(
            Path("inputs/bioconcepts2pubtatorcentral.gz"), pmids
        )

        # check if more than one abstract exists for year
        count = 0
        for sentence in tqdm.tqdm(sentences):
            if count > 0:
                break
            count += 1

        if count > 0:
            print("At least one abstract available")
            print("Creating word2vec model")
            model = Word2Vec(sentences, size=500, window=5, min_count=1, workers=4)
            model.save(
                Path(f"output/decades/models/word2vec_{str(year)}-{str(year+9)}.model")
            )
            print("Saved word2vec model")

            (
                similarity_score(model, pairs, years).to_csv(
                    f"outputs/decades/similarity_scores_{str(year)}-{str(year+9)}.tsv",
                    sep="\t",
                    index=False,
                )
            )

        else:
            print("No abstracts available")
    else:
        print("No PMIDs from this time period")
