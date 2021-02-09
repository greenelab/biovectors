"""
This module evluates word2vec's performance using AUROC and precision-recall.
"""
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_roc(labels, predictions, dummy_predictions):
    """
    Plots ROC curve.
    @param labels: numpy array of classes
    @param predictions: numpy array of corresponding word2vec similarity scores
    @param dummy_predictions: numpy array of corresponding dummy classifications
    """
    fp, tp, _ = roc_curve(labels, predictions)
    roc_auc = auc(fp, tp)

    fp_d, tp_d, _ = roc_curve(labels, dummy_predictions)
    roc_auc_d = auc(fp_d, tp_d)

    plt.figure()
    plt.plot(fp, tp, color="darkorange", lw=2, label="Word2vec, AUC = %0.2f" % roc_auc)
    plt.plot(
        fp_d,
        tp_d,
        color="navy",
        lw=2,
        label="Dummy Classifier, AUC = %0.2f" % roc_auc_d,
    )
    plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Analysis")
    plt.legend(loc="lower right")
    plt.savefig("figures/roc_curves.jpg")


def plot_precision_recall(labels, predictions):
    """
    Plots precision-recall curve.
    @param labels: numpy array of classes
    @param predictions: numpy array of corresponding similarity scores
    """
    precision, recall, threshold = precision_recall_curve(labels, predictions)
    avg_precision = average_precision_score(labels, predictions)

    plt.figure()
    plt.plot(
        recall, precision, label="Avg precision-recall score: %0.2f" % avg_precision
    )
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
    sorted_df = scores_df.sort_values(by=["score"], ascending=False).drop(
        scores_df.columns[:1], 1
    )
    sorted_df.to_csv("outputs/sorted_similarity_scores.tsv", sep="\t")


if __name__ == "__main__":
    base = os.path.abspath(os.getcwd())

    # word2vec
    scores_df = pd.read_csv(
        os.path.join(base, "outputs/similarity_scores.tsv"), sep="\t"
    )
    labels = np.array(scores_df[["class"]].values.tolist())
    predictions = np.array(scores_df[["score"]].values.tolist())

    # dummy
    dummy_df = pd.read_csv(os.path.join(base, "outputs/dummy_scores.tsv"), sep="\t")
    dummy_predictions = np.array(dummy_df[["dummy_score"]].values.tolist())

    # analysis
    plot_roc(labels, predictions, dummy_predictions)
    plot_precision_recall(labels, predictions)
    sort_similarity_scores(scores_df)
