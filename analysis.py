"""
This module evluates word2vec's performance using AUROC and precision-recall. 
"""
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_roc(labels, predictions, model):
    """
    Plots ROC curve.
    @param labels: numpy array of classes
    @param predictions: numpy array of corresponding similarity scores
    """
    fp, tp, _ = roc_curve(labels, predictions)
    roc_auc = auc(fp, tp)

    plt.figure()
    lw = 2
    plt.plot(fp, tp, 
             color="darkorange",
             lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model}: Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(f"figures/roc_curve_{model}.jpg")


def plot_precision_recall(labels, predictions):
    """
    Plots precision-recall curve.
    @param labels: numpy array of classes
    @param predictions: numpy array of corresponding similarity scores
    """
    precision, recall, threshold = precision_recall_curve(labels, predictions)
    avg_precision = average_precision_score(labels, predictions)

    plt.figure()
    plt.plot(recall, precision,
             label="Avg precision-recall score: %0.2f" % avg_precision)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Word2vec: Precision-Recall curve")
    plt.legend(loc="lower left")
    plt.savefig("figures/precision_recall_curve.jpg")


def sort_similarity_scores(scores_df):
    """
    Sorts similarity scores in descending order to differentiate TPs and FPs.
    @param scores_df: panda dataframe containing scores (see similarity_scores.tsv) 
    """
    sorted_df = scores_df.sort_values(by=["score"], ascending=False).drop(scores_df.columns[:1], 1)
    sorted_df.to_csv("outputs/sorted_similarity_scores.tsv", sep="\t")


if __name__ == "__main__":
    base = os.path.abspath(os.getcwd())

    # word2vec
    scores_df = pd.read_csv(
        os.path.join(base, "outputs/similarity_scores.tsv"),
        sep="\t"
    )
    labels = np.array(scores_df[["class"]].values.tolist())
    predictions = np.array(scores_df[["score"]].values.tolist())

    plot_roc(labels, predictions, "Word2vec")
    plot_precision_recall(labels, predictions)
    sort_similarity_scores(scores_df)

    # dummy
    dummy_df = pd.read_csv(
        os.path.join(base, "outputs/dummy_scores.tsv"),
        sep="\t"
    )
    dummy_predictions = np.array(dummy_df[["dummy_score"]].values.tolist())
    
    plot_roc(labels, dummy_predictions, "Dummy")

