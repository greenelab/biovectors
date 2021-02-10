"""
This module evaluates each decades's word2vec performance using AUROC and precision-recall.
Very similar to `analysis` module.
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


def plot_roc(labels, predictions, dummy_predictions, year):
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
    plt.title(f"ROC Curve Analysis ({str(year)}-{str(year+9)})")
    plt.legend(loc="lower right")
    plt.savefig(f"figures/analysis/roc_curves_{str(year)}-{str(year+9)}.jpg")


def plot_precision_recall(labels, predictions, year):
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
    plt.title(f"Word2vec: Precision-Recall curve ({str(year)}-{str(year+9)})")
    plt.legend(loc="lower left")
    plt.savefig(
        f"figures/analysis/precision_recall_curve_{str(year)}-{str(year+9)}.jpg"
    )


def sort_similarity_scores(scores_df, year):
    """
    Sorts similarity scores in descending order to differentiate TPs and FPs.
    @param scores_df: panda dataframe containing scores (see similarity_scores.tsv)
    """
    sorted_df = (
        scores_df.sort_values(by=["score"], ascending=False)
        #    .drop(scores_df.columns[:1], 0)
    )
    sorted_df.to_csv(
        f"outputs/decades/sorted/sorted_similarity_scores_{str(year)}-{str(year+9)}.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    base = os.path.abspath(os.getcwd())

    # iterate through results from 1971-2020 by decade
    years = [1971, 1981, 1991, 2001, 2011]
    for filename in os.listdir(os.path.join(base, "outputs/decades/")):
        for year in years:
            if f"similarity_scores_{str(year)}-{str(year+9)}" in filename:
                print(os.path.join(base, filename))
                # word2vec
                scores_df = pd.read_csv(
                    os.path.join(base, "outputs/decades/", filename), sep="\t"
                )
                labels = np.array(scores_df[["class"]].values.tolist())
                predictions = np.array(scores_df[["score"]].values.tolist())

                # dummy
                dummy_df = pd.read_csv(
                    os.path.join(
                        base,
                        f"outputs/decades/dummy_scores_{str(year)}-{str(year+9)}.tsv",
                    ),
                    sep="\t",
                )
                dummy_predictions = np.array(dummy_df[["dummy_score"]].values.tolist())

                # analysis
                plot_roc(labels, predictions, dummy_predictions, year)
                plot_precision_recall(labels, predictions, year)
                sort_similarity_scores(scores_df, year)
            else:
                continue
