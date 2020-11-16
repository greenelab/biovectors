"""
This module iterates through the pubtator data file to extract all unique PMIDs.
Unique PMIDs are written to a file.
"""
import os


def pmid(filename):
    """
    Extract unique PMIDs from pubtator data.
    @param filename: pubtator data file.
    """
    s = set()
    for line in open(filename, "rt"):
        if "|t|" in line:
            curr_pmid = line.split("|")[0]
            s.add(curr_pmid)
    return s


if __name__ == "__main__":
    base = os.path.abspath(os.getcwd())
    pmids = pmid(os.path.join(base, "inputs/bioconcepts2pubtatorcentral.gz"))

    with open(os.path.join(base, "outputs/pmid.tsv"), "w") as f:
        for pmid in pmids:
            f.write(pmid + "\n")
