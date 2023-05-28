import os

import numpy as np
import sklearn.metrics as m
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import src.evaluate.matching as matching_methods
from sklearn.metrics import ConfusionMatrixDisplay
import sklearn
import wandb
from PIL import Image
from src.evaluate import view_matches


class Metrics:
    def __init__(self, method_name, dataset_name, Fq, Fm, GT, GTsoft=None, matching_method=None, rootdir=None, q_pths=None, db_pths=None):
        wandb.login()
        self.run = wandb.init(
            project="VPR-Metrics",
            name=method_name,
            tags=[method_name, dataset_name, 'GTsoft' if isinstance(GTsoft, type(np.ones(1))) else 'GThard'],

            config={
                'method': method_name,
                'dataset': dataset_name,
                'GT_type': 'GTsoft' if isinstance(GTsoft, type(np.ones(1))) else 'GThard',
                'session_type': 'single-session' if Fq.all() == Fm.all() else 'multi-session'
            }
        )

        self.rootdir = rootdir
        self.q_pths = q_pths
        self.db_pths = db_pths
        self.method_name = method_name
        self.dataset_name = dataset_name
        self.Fq = Fq
        self.Fm = Fm
        if matching_method:
            self.S = matching_method(Fq, Fm)
        else:
            self.S = cosine_similarity(Fq, Fm)
        self.GT = GT
        self.GTsoft = GTsoft

    def log_metrics(self, matching='multi', threshold_type='single'):
        prec = self.precision(matching=threshold_type)
        recall = self.recall(matching=threshold_type)
        recallAt1 = self.recallAtK(K=1)
        recallAt5 = self.recallAtK(K=5)
        recallAt10 = self.recallAtK(K=10)
        recallAt100precision = self.recallAt100precision(matching='multi')
        auprc = self.AU_PRC(matching=matching)

        # plot curves
        self.ROCcurve(matching=matching)
        self.PRcurve(matching=matching)
        self.confusion_matrix(matching='single')

        # Visualize the matches
        if self.q_pths is not None and self.db_pths is not None:
            self.view_matches(self.q_pths, self.db_pths, self.GT, self.S,
                              self.dataset_name, self.method_name,
                              matching=threshold_type, GTsoft=self.GTsoft)

        metrics = {"method": self.method_name,
                   "dataset": self.dataset_name,
                   "gt_type": 'GTsoft' if isinstance(self.GTsoft, type(np.ones(1))) else 'GThard',
                   "session_type": 'single-session' if self.Fq.all() == self.Fm.all() else 'multi-session',
                   "precision": [prec],
                   "recall": [recall],
                   "recall@1": [recallAt1],
                   "recall@5": [recallAt5],
                   "recall@10": [recallAt10],
                   "recall@100precision": [recallAt100precision],
                   "auprc": [auprc]}
        print(metrics)
        metrics_table = wandb.Table(dataframe=pd.DataFrame.from_dict(metrics))
        self.run.log({"metrics": metrics_table})
        wandb.finish()

    def view_matches(self, q_pths, db_pths, GT, S, dataset, method, matching='single', GTsoft=None, show=False):
        if matching == 'single':
            M = matching_methods.best_match_per_query(S)
        elif matching == 'auto':
            M = matching_methods.thresholding(S, 'auto')
        elif type(matching) == float:
            M = matching_methods.thresholding(S, matching)

        TP = []
        FP = []

        GT = GTsoft if GTsoft else GT

        for i in GT.shape[0]:
            for j in GT.shape[1]
                if GT[i, j] == 0 and M[i, j] == 1:
                    FP.append([j, i])
                if GT[i, j] == 1 and M[i, j] == 1:
                    TP.append([j, i])

        TP, FP = np.array(TP), np.array(FP)

        img = view_matches.show(db_pths, q_pths, TP, FP, show=show)

        wandb.log({'matches_' + self.dataset_name: wandb.Image(img)})

    def precision(self, matching='single'):
        if matching == 'single':
            M = matching_methods.best_match_per_query(self.S)
        elif matching == 'auto':
            M = matching_methods.thresholding(self.S, 'auto')
        elif type(matching) == float:
            M = matching_methods.thresholding(self.S, matching)
        return sklearn.metrics.precision_score(self.GTsoft.flatten().astype(int), M.flatten().astype(int))

    def confusion_matrix(self, matching='single'):
        if matching == 'single':
            M = matching_methods.best_match_per_query(self.S)
        elif matching == 'auto':
            M = matching_methods.thresholding(self.S, 'auto')
        elif type(matching) == float:
            M = matching_methods.thresholding(self.S, matching)

        y_truth = self.GTsoft if isinstance(self.GTsoft, type(np.ones(1))) else self.GThard

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cm = sklearn.metrics.confusion_matrix(y_truth.flatten().astype(int), M.flatten().astype(int))
        cm = ConfusionMatrixDisplay.from_predictions(y_truth.flatten().astype(int), M.flatten().astype(int),
                                                     display_labels=['0', '1'], ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['No Match', 'Place Match'])
        ax.yaxis.set_ticklabels(['No Match', 'Place Match'])
        cm.plot()

        # labels, title and ticks
        fig.savefig(self.rootdir + '/src/evaluate/tmp/plot.png')
        plot = Image.open(self.rootdir + '/src/evaluate/tmp/plot.png')
        wandb.log({'confmap_' + self.dataset_name: wandb.Image(plot)})

    def recall(self, matching='single'):
        if matching == 'single':
            M = matching_methods.best_match_per_query(self.S)
        elif matching == 'auto':
            M = matching_methods.thresholding(self.S, 'auto')
        elif type(matching) == float:
            M = matching_methods.thresholding(self.S, matching)
        return sklearn.metrics.recall_score(self.GT.flatten().astype(int), M.flatten().astype(int))


    def createPR(self, matching='multi', n_thresh=100):
        """
        Calculates the precision and recall at n_thresh equally spaced threshold values
        for a given similarity matrix S_in and ground truth matrices GThard and GTsoft for
        single-best-match VPR or multi-match VPR.

        The matrices S_in, GThard and GTsoft are two-dimensional and should all have the
        same shape.
        The matrices GThard and GTsoft should be binary matrices, where the entries are
        only zeros or ones.
        The matrix S_in should have continuous values between -Inf and Inf. Higher values
        indicate higher similarity.
        The string matching should be set to either "single" or "multi" for single-best-
        match VPR or multi-match VPR.
        The integer n_tresh controls the number of threshold values and should be >1.
        """

        assert (self.S.shape == self.GT.shape), "S_in, GThard and GTsoft must have the same shape"
        assert (self.S.ndim == 2), "S_in, GThard and GTsoft must be two-dimensional"
        assert (n_thresh > 1), "n_thresh must be >1"

        # ensure logical datatype in GT and GTsoft
        GT = self.GT.astype('bool')

        # copy S and set elements that are only true in GTsoft to min(S) to ignore them during evaluation
        S = self.S.copy()
        if self.GTsoft is not None:
            S[self.GTsoft & ~self.GT] = S.min()

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

        # select start and end threshold
        startV = S.max()  # start-value for threshold
        endV = S.min()  # end-value for threshold

        # iterate over different thresholds
        for i in np.linspace(startV, endV, n_thresh):
            B = S >= i  # apply threshold

            TP = np.count_nonzero(GT & B)  # true positives
            FP = np.count_nonzero((~GT) & B)  # false positives

            P.append(TP / (TP + FP))  # precision
            R.append(TP / GTP)  # recall
        return P, R

    def ROCcurve(self, matching='multi', n_thresh=100):
        assert (self.S.shape == self.GT.shape), "S_in, GThard and GTsoft must have the same shape"
        assert (self.S.ndim == 2), "S_in, GThard and GTsoft must be two-dimensional"
        assert (n_thresh > 1), "n_thresh must be >1"

        # ensure logical datatype in GT and GTsoft
        GT = self.GT.astype('bool')

        # copy S and set elements that are only true in GTsoft to min(S) to ignore them during evaluation
        S = self.S.copy()
        if self.GTsoft is not None:
            S[self.GTsoft & ~self.GT] = S.min()

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

        scores = np.concatenate((1 - S.flatten()[:, None], S.flatten()[:, None]), axis=1)
        wandb.log({"roc_" + self.dataset_name: wandb.plot.roc_curve(GT.flatten().astype(int),
                                                                    scores, labels=["No_Match", "Place_Match"],
                                                                    classes_to_plot=[1],
                                                                    title="ROC Curve - Dataset: " + self.dataset_name)})

    def PRcurve(self, matching='multi', n_thresh=100):
        assert (self.S.shape == self.GT.shape), "S_in, GThard and GTsoft must have the same shape"
        assert (self.S.ndim == 2), "S_in, GThard and GTsoft must be two-dimensional"
        assert (n_thresh > 1), "n_thresh must be >1"

        # ensure logical datatype in GT and GTsoft
        GT = self.GT.astype('bool')

        # copy S and set elements that are only true in GTsoft to min(S) to ignore them during evaluation
        S = self.S.copy()
        if self.GTsoft is not None:
            S[self.GTsoft & ~self.GT] = S.min()

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

        scores = np.concatenate((1 - S.flatten()[:, None], S.flatten()[:, None]), axis=1)
        wandb.log({"pr_" + self.dataset_name: wandb.plot.pr_curve(GT.flatten().astype(int), scores,
                                                                  labels=["No_Match", "Place_Match"],
                                                                  classes_to_plot=[1],
                                                                  title="PR Curve - Dataset: " + self.dataset_name)})

    def recallAtK(self, K=1):
        """
        Calculates the recall@K for a given similarity matrix S_in and ground truth matrices
        GThard and GTsoft.

        The matrices S_in, GThard and GTsoft are two-dimensional and should all have the
        same shape.
        The matrices GThard and GTsoft should be binary matrices, where the entries are
        only zeros or ones.
        The matrix S_in should have continuous values between -Inf and Inf. Higher values
        indicate higher similarity.
        The integer K>=1 defines the number of matching candidates that are selected and
        that must contain an actually matching image pair.
        """
        assert (self.S.shape == self.GT.shape), "S_in and GThard must have the same shape"
        assert (self.S.ndim == 2), "S_in, GThard and GTsoft must be two-dimensional"
        assert (K >= 1), "K must be >=1"

        # ensure logical datatype in GT and GTsoft
        GT = self.GT.astype('bool')
        GTsoft = self.GTsoft.astype('bool')

        # copy S and set elements that are only true in GTsoft to min(S) to ignore them during evaluation
        S = self.S.copy()
        S[self.GTsoft & ~self.GT] = S.min()

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

    def recallAt100precision(self, matching='multi', n_thresh=100):
        """
        Calculates the maximum recall at 100% precision for a given similarity matrix S_in
        and ground truth matrices GThard and GTsoft for single-best-match VPR or multi-match
        VPR.

        The matrices S_in, GThard and GTsoft are two-dimensional and should all have the
        same shape.
        The matrices GThard and GTsoft should be binary matrices, where the entries are
        only zeros or ones.
        The matrix S_in should have continuous values between -Inf and Inf. Higher values
        indicate higher similarity.
        The string matching should be set to either "single" or "multi" for single-best-
        match VPR or multi-match VPR.
        The integer n_tresh controls the number of threshold values during the creation of
        the precision-recall curve and should be >1.
        """

        assert (self.S.shape == self.GT.shape), "S_in and GThard must have the same shape"
        assert (self.S.ndim == 2), "S_in, GThard and GTsoft must be two-dimensional"
        assert (matching in ['single',
                             'multi']), "matching should contain one of the following strings: [single, multi]"
        assert (n_thresh > 1), "n_thresh must be >1"

        # get precision-recall curve
        P, R = self.createPR(matching=matching, n_thresh=n_thresh)
        P = np.array(P)
        R = np.array(R)

        # recall values at 100% precision
        R = R[P == 1]

        # maximum recall at 100% precision
        R = R.max()

        return R

    def AU_PRC(self, matching='single', n_thresh=100):
        P, R = self.createPR(matching=matching, n_thresh=n_thresh)
        return np.trapz(P, R)

    def descriptor_size(self):
        type = str(self.Fm.dtype)
        nbytes = self.Fm[0].nbytes
        size = self.Fm.shape[1]
        print("Descriptor has size: ", size, " of type ", type, "taking up ", nbytes, " bytes")
