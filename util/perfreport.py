#!/usr/bin/env python
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from pandas_ml import ConfusionMatrix
import pickle
import numpy as np


def generate_metrics(y_true, y_pred, scores, class_labels):
    # One-hot encode the truth (for multiclass metrics, if needed)
    y_true_onehot = label_binarize(y_true, classes=class_labels)
    m = {}
    m["sklearn"] = {}
    m["pandas-ml"] = {}

    # Calculate accuracy
    m["sklearn"]["acc"] = metrics.accuracy_score(y_true, y_pred)

    # Confusion matrix
    m["sklearn"]["confmat"] = metrics.confusion_matrix(
        y_true, y_pred, labels=class_labels
    )

    # Generate classification report
    m["sklearn"]["report"] = metrics.classification_report(
        y_true, y_pred, target_names=class_labels
    )

    # Get AUCs
    auc_indiv = metrics.roc_auc_score(y_true_onehot, scores, average=None)
    m["sklearn"]["auc_indiv"] = auc_indiv
    m["sklearn"]["auc_avg"] = np.mean(auc_indiv)

    # Get pandas-ml metrics
    m["pandas-ml"]["cm"] = ConfusionMatrix(y_true, y_pred, labels=class_labels)

    return m


def print_metrics(results_file):
    # Read data
    with open(results_file, "rb") as f:
        data = pickle.load(f)

    # Pull out data
    class_labels = data["class_labels"]
    y_pred = data["y_pred"]
    y_true = data["y_true"]
    scores = data["scores"]

    m = generate_metrics(y_true, y_pred, scores, class_labels)

    # Print everything
    print("SCIKIT-LEARN METRICS:\n")
    print("Accuracy: {:0.3f}".format(m["sklearn"]["acc"]))

    print("\nConfusion Matrix:")
    print(m["sklearn"]["confmat"])

    print("\nClassification Report:")
    print(m["sklearn"]["report"])

    print("\nAUC:")
    for label, score in zip(class_labels, m["sklearn"]["auc_indiv"]):
        print("{}\t{:0.3f}".format(label, score))
    print("AVG:\t{:0.3f}".format(m["sklearn"]["auc_avg"]))

    print("\n\nPANDAS-ML STATS:\n")
    m["pandas-ml"]["cm"].print_stats()
