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


def curvepr(GT, S, GTsoft = None, n_thresh=100, matching: str = 'multi') -> tuple[np.ndarray, np.ndarray]:
    assert S.shape == GT.shape, "S and GT must be the same shape"
    assert (S.ndim == 2), "S_in, GThard and GTsoft must be two-dimensional"
    GT = GT.astype('bool')
    # copy S and set elements that are only true in GTsoft to min(S) to ignore them during evaluation
    S = S.copy()
    if GTsoft is not None:
        S[GTsoft & ~GT] = S.min()
    # single-best-match or multi-match VPR
    if matching == 'single':
        # count the number of ground-truth positives (GTP)
        GTP = np.count_nonzero(GT.any(0))
        # GT-values for best match per query (i.e., per column)
        GT = GT[np.argmax(S, axis=0), np.arange(GT.shape[1])]
        # similarities for best match per query (i.e., per column)
        S = np.max(S, axis=0)
    elif matching == 'multi':
        # count the number of ground-truth positives (GTP)
        GTP = np.count_nonzero(GT)  # ground truth positive
        # init precision and recall vectors
    R = [0, ]
    P = [1, ]

    # select start and end treshold
    startV = S.max()  # start-value for treshold
    endV = S.min()  # end-value for treshold

    # iterate over different thresholds
    for i in np.linspace(startV, endV, n_thresh):
        B = S >= i  # apply threshold

        TP = np.count_nonzero(GT & B)  # true positives
        FP = np.count_nonzero((~GT) & B)  # false positives

        P.append(TP / (TP + FP))  # precision
        R.append(TP / GTP)  # recall

    return P, R



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
    P, R = curvepr(GT, S)
    # recall values at 100% precision
    R = R[P == 1]
    # maximum recall at 100% precision
    R = R.max()
    return R


def recallAtNprecision(GT, S, N) -> float:
    assert (S.shape == GT.shape)
    assert (S.ndim == 2)
    P, R = curvepr(GT, S)
    idx = np.argwhere(P > N).min()
    return P[idx]


def average_precision(GT, S):
    assert (S.shape == GT.shape)
    assert (S.ndim == 2)
    return average_precision_score(GT.flatten().astype(int), S.flatten())


# ================================= Plots =============================================================================

def plot_curvepr(GT: np.ndarray, S_data: dict, dataset_name=None, show=False, matching: str = 'multi', GTsoft = None) -> None:
    fig, ax = plt.subplots()
    for name, S in S_data.items():
        P, R = curvepr(GT, S, GTsoft=GTsoft, matching=matching)
        ax.plot(P, R, label=name)

    plt.legend()
    ax.set_title("PR Curve for " + dataset_name)
    pth = config.root_dir + '/vpr/evaluate/figures/pr_curves/'
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.01, 1), plt.ylim(0, 1.01)
    if not os.path.exists(pth):
        os.makedirs(pth)

    fig.savefig(pth + dataset_name + '.png')
    if show:
        plt.show()
    plt.close()
    return 0


def plot_recallat1(GT: np.ndarray, S_data: dict, dataset_name=None, show=False, matching: str = 'multi', GTsoft=None) -> None:
    scores, names = [], []
    for name, S in S_data.items():
        scores.append(recallAtK(GT, S, 1))
        names.append(name)

    print("===========================", scores)
    fig, ax = plt.subplots()
    ax.bar(names, scores)
    ax.set_xticklabels(names, rotation=45)
    ax.set_title("Recall@1 for " + dataset_name)
    pth = config.root_dir + '/vpr/evaluate/figures/recall@1_plot/'
    if not os.path.exists(pth):
        os.makedirs(pth)

    plt.tight_layout()

    fig.savefig(pth + dataset_name + '.png')

    if show:
        plt.show()
    return 0


def plot_average_precision(GT: np.ndarray, S_data: dict, dataset_name=None, show=False, matching: str = 'multi', GTsoft=None) -> None:
    scores, names = [], []
    for name, S in S_data.items():
        scores.append(average_precision(GT, S))
        names.append(name)
    fig, ax = plt.subplots()
    ax.bar(names, scores)
    ax.set_xticklabels(names, rotation=45)
    ax.set_title("Recall@1 for " + dataset_name)
    pth = config.root_dir + '/vpr/evaluate/figures/average_precision/'
    if not os.path.exists(pth):
        os.makedirs(pth)

    plt.tight_layout()

    fig.savefig(pth + dataset_name + '.png')

    if show:
        plt.show()
    return 0

def plot_precision(GT: np.ndarray, S_data: dict, dataset_name=None, show=False, matching: str = 'multi', GTsoft=None) -> None:
    scores, names = [], []
    for name, S in S_data.items():
        scores.append(precision(GT, S, threshold_type='single'))
        names.append(name)
    fig, ax = plt.subplots()
    ax.bar(names, scores)
    ax.set_xticklabels(names, rotation=45)
    ax.set_title("Recall@1 for " + dataset_name)
    pth = config.root_dir + '/vpr/evaluate/figures/precision/'
    if not os.path.exists(pth):
        os.makedirs(pth)

    plt.tight_layout()

    fig.savefig(pth + dataset_name + '.png')

    if show:
        plt.show()
    return 0





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
