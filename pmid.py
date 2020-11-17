"""
This module iterates through the pubtator data file to extract all unique PMIDs.
Unique PMIDs are written to a file.
"""
import pandas as pd
import os
import gzip


def pmid(filename):
    """
    Extract unique PMIDs from pubtator data.
    @param filename: pubtator data file.
    """
    s = set()
    for line in gzip.open(filename, "rt"):
        if "|t|" in line:
            curr_pmid = line.split("|")[0]
            s.add(curr_pmid)
    return s


if __name__ == "__main__":
    base = os.path.abspath(os.getcwd())
    pmids = pmid(os.path.join(base, "inputs/bioconcepts2pubtatorcentral.gz"))

    df = pd.DataFrame(columns=["pmid"])
    for pmid in pmids:
        df = df.append(pmid, ignore_index=True)
    
    df.to_csv(
        os.path.join(base, "outputs/pmids.tsv"), 
        sep="\t", 
        index=False,
        compression="xz"
    )
