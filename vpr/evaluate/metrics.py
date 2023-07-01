import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_curve, precision_recall_curve, average_precision_score
import config
from vpr.evaluate import matching


def precision(GT, S, threshold_type: str = 'multi'):
    if threshold_type == 'single':
        P = matching.best_match_per_query(S)
    elif threshold_type == 'multi':
        P = matching.thresholding(S, 'auto')
    else:
        raise Exception("threshold type should be 'single' or 'multi'")
    return precision_score(GT.flatten().astype(int), P.flatten().astype(int))


def recall(GT, S, threshold_type: str = 'multi'):
    # Compute the predictions from similarity matrix
    if threshold_type == 'single':
        P = matching.best_match_per_query(S)
    elif threshold_type == 'multi':
        P = matching.thresholding(S, 'auto')
    return recall_score(GT.flatten().astype(int), P.flatten().astype(int))


def curvepr(GT, S) -> tuple[np.ndarray, np.ndarray]:
    assert (S.shape == GT.shape), "S_in, GThard and GTsoft must have the same shape"
    assert (S.ndim == 2), "S_in, GThard and GTsoft must be two-dimensional"
    precision, recall, thresholds = precision_recall_curve(GT.flatten().astype(int), S.flatten())
    return precision, recall, thresholds


def curveroc(GT, S) -> None:
    assert (S.shape == GT.shape), "S_in, GThard and GTsoft must have the same shape"
    assert (S.ndim == 2), "S_in, GThard and GTsoft must be two-dimensional"
    fpr, tpr, thresholds = roc_curve(GT.flatten().astype(int), S.flatten())
    return fpr, tpr, thresholds


def recallAtK(GT, S, K: int = 1) -> float:
    assert (S.shape == GT.shape), "S_in and GThard must have the same shape"
    assert (S.ndim == 2), "S_in, GThard and GTsoft must be two-dimensional"
    assert (K >= 1), "K must be >=1"
    # ensure logical datatype in GT and GTsoft
    GT = GT.astype('bool')
    # copy S and set elements that are only true in GTsoft to min(S) to ignore them during evaluation
    S = S.copy()
    # discard all query images without an actually matching database image
    j = GT.sum(0) > 0  # columns with matches
    S = S[:, j]  # select columns with a match
    GT = GT[:, j]  # select columns with a match
    # select K highest similarities
    i = S.argsort(0)[-K:, :]
    j = np.tile(np.arange(i.shape[1]), [K, 1])
    GT = GT[i, j]
    # recall@K
    RatK = np.sum(GT.sum(0) > 0) / GT.shape[1]
    return RatK


def recallAt100precision(GT, S) -> float:
    assert (S.shape == GT.shape), "S_in and GThard must have the same shape"
    assert (S.ndim == 2), "S_in, GThard and GTsoft must be two-dimensional"
    # get precision-recall curve
    P, R, _ = curvepr(GT, S)
    # recall values at 100% precision
    R = R[P == 1]
    # maximum recall at 100% precision
    R = R.max()
    return R


def recallAtNprecision(GT, S, N) -> float:
    assert (S.shape == GT.shape)
    assert (S.ndim == 2)
    P, R, _ = curvepr(GT, S)
    idx = np.argwhere(P > N).min()
    return P[idx]


def average_precision(GT, S):
    assert (S.shape == GT.shape)
    assert (S.ndim == 2)
    return average_precision_score(GT.flatten().astype(int), S.flatten())


# ================================= Plots =============================================================================

def plot_curvepr(GT: np.ndarray, S_data: dict, dataset_name=None, show=False) -> None:
    for name, S in S_data.items():
        P, R, _ = curvepr(GT, S)
        plt.plot(P, R, label=name)

    plt.legend()
    plt.title("PR Curve for " + dataset_name)
    pth = config.root_dir + '/vpr/evaluate/figures/pr_curves/'
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    if not os.path.exists(pth):
        os.makedirs(pth)
    if show:
        plt.show()
    plt.savefig(pth + dataset_name + '.png')
    return 0


def plot_curveroc(GT: np.ndarray, S_data: dict, dataset_name=None, show=False) -> None:
    for name, S in S_data.items():
        fpr, tpr, _ = curveroc(GT, S)
        plt.plot(fpr, tpr, label=name)

    plt.title("ROC Curve for " + dataset_name)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    pth = config.root_dir + '/vpr/evaluate/figures/pr_curves/'
    if not os.path.exists(pth):
        os.makedirs(pth)
    if show:
        plt.show()
    plt.savefig(pth + dataset_name + '.png')


def compute_metrics(GT, S_data, dataset_name=None):
    pth = config.root_dir + '/vpr/evaluate/figures/metrics/'
    if not os.path.exists(pth):
        os.makedirs(pth)

    precisions = []
    recalls = []
    average_precisions = []
    recallAt100precisions = []
    recallat90precisions = []
    recallat5 = []
    recallat10 = []
    recallat1 = []
    names = []
    for name, S in S_data.items():
        precisions.append(precision(GT, S))
        recalls.append(recall(GT, S))
        average_precisions.append(average_precision(GT, S))
        recallAt100precisions.append(recallAt100precision(GT, S))
        recallat90precisions.append(recallAtNprecision(GT, S, 0.9))
        recallat5.append(recallAtK(GT, S, 5))
        recallat10.append(recallAtK(GT, S, 10))
        recallat1.append(recallAtK(GT, S, 1))
        names.append(name)

    data_dict = {"precision": precisions,
                 "recall": recalls,
                 "average_precision": average_precisions,
                 "recallAt100p": recallAt100precisions,
                 "recallAt90p": recallat90precisions,
                 "recallAt10": recallat10,
                 "recallAt5": recallat5,
                 "recallAt1": recallat1,
                 "method": names}

    df = pd.DataFrame.from_dict(data_dict)
    df.set_index('method')
    if os.path.exists(config.root_dir + '/vpr/evaluate/figures/metrics/'):
        os.makedirs(config.root_dir + '/vpr/evaluate/figures/metrics/')
    df.to_csv(config.root_dir + '/vpr/evaluate/figures/metrics/' + dataset_name + '.csv')
    return df
