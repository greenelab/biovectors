import gzip
import random

import pandas as pd


class SentencesIterator:
    """
    Extracts title + abstracts from Pubtator data. Replaces any instance of a
    gene or disease with the Entrez gene ID or MESH ID, respectively.
    """

    def __init__(self, filename, pmids=None):
        self.filename = filename
        self.pmids = pmids

    def __iter__(self):
        print("Getting sentences...")
        pmid_to_check = curr_pmid = curr_title = curr_abstract = curr_total = None

        for line in gzip.open(self.pubtator_filename, "rt"):
            if "|t|" in line or "|a|" in line:
                pmid_to_check = line.split("|")[0]
            elif line.strip() != "":
                pmid_to_check = line.split("\t")[0]

            if pmid_to_check is not None:
                if self.pmids is None or int(pmid_to_check) in self.pmids:
                    if "|t|" in line:
                        curr_title = line.split("|")[2]
                        continue

                    if "|a|" in line:
                        if curr_total is not None:
                            yield curr_total.split()
                        curr_pmid = line.split("|")[0]
                        curr_abstract = line.split("|")[2]
                        if curr_title is not None:
                            curr_total = (
                                curr_title + curr_abstract
                            )  # combine title and abstract
                        continue

                    else:
                        if curr_pmid is not None and curr_total is not None:
                            if (
                                "Disease" in line or "Gene" in line
                            ):  # targeting gene-disease pairs
                                features = line.split("\t")
                                if (
                                    features[0] == curr_pmid
                                    and features[5].strip() != ""
                                ):
                                    curr_total = curr_total.replace(
                                        features[3], features[5].strip(), 1
                                    )

        if curr_total is not None:
            yield curr_total.split()


def get_gene_disease_pairs(gene_disease_filename, do_mesh_filename):
    """
    Extracts hetionet gene-disease pairs and generates negative pairs by randomizing positive pairs.
    @param gene_disease_filename: file containing hetionet gene disease pairs
    @param do_mesh_filename: file containing corresponding doid and mesh ids (because hetnet
        contains only doids)
    @return gene-disease pairs
    """
    random.seed(100)  # reproducibility
    gene_disease_df = pd.read_csv(gene_disease_filename, sep="\t")
    do_mesh_df = pd.read_csv(do_mesh_filename, sep="\t")

    # create doid-mesh list
    do_mesh_pairs = dict(zip(do_mesh_df.doid_code, "MESH:" + do_mesh_df.mesh_id))
    gene_disease_df["mesh_id"] = gene_disease_df["doid_id"].replace(do_mesh_pairs)
    # remove rows that don't have a DOID-MESH id mapping
    # gene_disease_df = gene_disease_df.query("~mesh_id.str.contains('DOID:')")
    gene_disease_df = gene_disease_df[~gene_disease_df.mesh_id.str.contains("DOID:")]
    # get positive pairs
    positive_pairs = gene_disease_df[["mesh_id", "entrez_gene_id"]].values.tolist()

    # randomize pairings to create negative pairs
    gene_disease_df["random_gene"] = random.sample(
        gene_disease_df["entrez_gene_id"].values.tolist(),
        len(gene_disease_df["entrez_gene_id"].values.tolist()),
    )
    randomized_pairs = gene_disease_df[["mesh_id", "random_gene"]].values.tolist()
    negative_pairs = []
    for pair in random.sample(randomized_pairs, len(randomized_pairs)):
        if pair not in positive_pairs:
            negative_pairs.append(pair)

    # append class to each pair
    for pair in positive_pairs:
        pair.append(1)
    for pair in negative_pairs:
        pair.append(0)
    gene_disease_pairs = positive_pairs + negative_pairs

    return gene_disease_pairs


def similarity_scores(model, pairs):
    """
    Computes cosine similarity between gene and disease if both exist in the word2vec vocabulary.
    Outputs similarity_scores.tsv.
    @param model: the trained word2vec model
    @param pairs: all gene-disease pairs to be tested
    """
    similarity_scores_df = pd.DataFrame(columns=["disease", "gene", "class", "score"])
    for pair in pairs:
        if all(str(vocab) in model.wv.vocab for vocab in pair[:2]):
            score = model.wv.similarity(str(pair[0]), str(pair[1]))
            new_row = {
                "disease": pair[0],
                "gene": pair[1],
                "class": pair[2],
                "score": score,
            }
            similarity_scores_df = similarity_scores_df.append(
                new_row, ignore_index=True
            )

    return similarity_scores_df
