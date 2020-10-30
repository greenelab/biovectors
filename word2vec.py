from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics import roc_curve, roc_auc_score, auc
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt


class Sentences(object):
    """
    Extracts title + abstracts from Pubtator data. Replaces any instance of a 
    gene or disease with the Entrez gene ID or MESH ID, respectively. 
    """
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        curr_pmid = curr_title = curr_abstract = curr_total = None
        for line in open(self.filename):
            if "|t|" in line:
                curr_title = line.split("|")[2]
                continue
            if "|a|" in line:
                if curr_total is not None:
                    yield curr_total.split()
                curr_pmid = line.split("|")[0]
                curr_abstract = line.split("|")[2]
                if curr_title is not None:
                    curr_total = curr_title + curr_abstract  # combine title and abstract
                continue
            else:
                if curr_pmid is not None and curr_total is not None:
                    if "Disease" in line or "Gene" in line:  # targeting gene-disease pairs
                        features = line.split("\t")
                        if int(features[0]) == int(curr_pmid) and features[5].strip() != "":
                            curr_total = curr_total.replace(features[3], features[5].strip(), 1)
        yield curr_total.split()
                    

def create_word2vec(sentences):
    """
    Creates a word2vec model.
    @param sentences: list of list of words in each sentence (title + abstract).
    """
    model = Word2Vec(sentences, size=500, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    return model


def get_gene_disease_pairs(gene_disease_filename, do_mesh_filename):
    """
    Extracts hetionet gene-disease pairs and generates negative pairs by randomizing positive pairs. 
    @param gene_disease_filename: file containing hetionet gene disease pairs
    @param do_mesh_filename: file containing corresponding doid and mesh ids (because hetnet
        contains only doids)
    """
    random.seed(100)  # reproducibility
    gene_disease_df = pd.read_csv(gene_disease_filename, sep="\t")
    do_mesh_df = pd.read_csv(do_mesh_filename, sep="\t")

    # create doid-mesh list
    do_mesh_pairs = dict(zip(do_mesh_df.doid_code, "MESH:"+do_mesh_df.mesh_id))
    gene_disease_df["mesh_id"] = gene_disease_df["doid_id"].replace(do_mesh_pairs) 
    # remove rows that don't have a DOID-MESH id mapping
    gene_disease_df = gene_disease_df[~gene_disease_df.mesh_id.str.contains("DOID:")]
    # get positive pairs
    positive_pairs = gene_disease_df[["mesh_id", "entrez_gene_id"]].values.tolist()
    
    # randomize pairings to create negative pairs
    gene_disease_df["random_gene"] = random.sample(gene_disease_df["entrez_gene_id"].values.tolist(), len(gene_disease_df["entrez_gene_id"].values.tolist()))
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


def get_scores(model, pairs):
    """
    Computes cosine similarity between gene and disease if both exist in the word2vec vocabulary.
    @param model: the trained word2vec model
    @param pairs: all gene-disease pairs to be tested
    """
    similarity_scores_df = pd.DataFrame(columns=["disease", "gene", "class", "score"]) 
    for pair in pairs:
        if all(str(vocab) in model.wv.vocab for vocab in pair[:2]):
            score = model.wv.similarity(str(pair[0]), str(pair[1]))
            new_row = {"disease": pair[0], "gene": pair[1], "class": pair[2], "score": score}
            similarity_scores_df = similarity_scores_df.append(new_row, ignore_index=True)
    similarity_scores_df.to_csv("similarity_scores.tsv", sep="\t")


def calc_roc(scores_filename):
    """
    Generates ROC curve.
    @param scores_filename: file containg similarity scores
    """
    df = pd.read_csv(scores_filename, sep="\t")
    labels = np.array(df[["class"]].values.tolist())
    predictions = np.array(df[["score"]].values.tolist())
    return(roc_curve(labels, predictions))


if __name__ == "__main__":
    base = os.path.abspath(os.getcwd())
    # only need the 3 lines below to run on pmacs cluster
    sentences = Sentences(os.path.join(base, "data/testdata_pubtator_central_export.pubtator"))
    word2vec = create_word2vec(sentences)
    pairs = get_gene_disease_pairs(os.path.join(base, "data/hetnet_gene_disease_pairs.tsv"), os.path.join(base, "data/DO-slim-to-mesh.tsv"))
    get_scores(word2vec, pairs)

    # roc
    fp, tp, _ = calc_roc(os.path.join(base, "similarity_scores.tsv"))
    roc_auc = auc(fp, tp)

    # plot roc
    plt.figure()
    lw = 2
    plt.plot(fp, tp, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Word2vec: Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
