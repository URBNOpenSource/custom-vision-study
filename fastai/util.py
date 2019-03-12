#!/usr/bin/env python
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import pickle
import numpy as np
import pandas as pd
from pprint import pprint


def evaluate_data(learn, dataset="test"):
    # Evaluate test data
    is_test = True if dataset.lower() == "test" else False
    yy = learn.get_preds(is_test=is_test)
    scores = np.array(yy[0])

    # Get class labels
    class_labels = learn.data.classes

    # Assign predicted labels from fastai
    y_pred_num = np.array(yy[0].max(1)[1])
    y_pred = [class_labels[z] for z in y_pred_num]

    return y_pred, scores, class_labels


def get_matching_truth(labels_file, data, truth_set="test"):
    # Get true label list
    df = pd.read_csv(labels_file, header=None, names=["name", "value"])

    keep_inds = df.name.str.contains(truth_set)
    df_keep = df[keep_inds]
    df_keep["name"] = [d.split("/")[-1] for d in df_keep.name.tolist()]
    df_keep.reset_index(inplace=True, drop=True)

    # Get names of test data from fastai.
    truth_set == truth_set.lower()
    if truth_set == "test":
        ds = data.test_ds.ds
    elif truth_set == "val":
        ds = data.valid_ds.ds
    else:
        ds = data.train_ds.ds
    eval_filenames = [str(d).split("/")[-1] for d in ds.x]

    # Get truth fields, indexed on those from fastai
    true_labels = [df_keep[df_keep.name == t]["value"].iloc[0] for t in eval_filenames]
    y_true = [str(z) for z in true_labels]

    return y_true


def eval_rollup(labels_file, learn, evalset="test"):
    # Get predictions
    y_pred, scores, class_labels = evaluate_data(learn, dataset=evalset)

    # Get truth
    y_true = get_matching_truth(labels_file, learn.data, truth_set=evalset)

    # Calculate results
    results = generate_metrics(y_true, y_pred, scores, class_labels)

    # Print results
    print("Dataset: {}".format(evalset))
    pprint(results)

    return results, y_true, y_pred, scores, class_labels


def generate_metrics(y_true, y_pred, scores, class_labels):
    # One-hot encode the truth (for multiclass metrics, if needed)
    y_true_onehot = label_binarize(y_true, classes=class_labels)
    m = {}
    m["sklearn"] = {}

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
