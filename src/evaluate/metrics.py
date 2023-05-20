import numpy as np
import sklearn.metrics as m
import matplotlib.pyplot as plt
import pandas as pd
import src.evaluate.matching as matching_methods
import sklearn
import wandb

class Metrics:
    def __init__(self, method, Fq, Fm, GT, GTsoft=None, save_results=True, save_path=None):
        wandb.login()
        wandb.init(
            project="VPR-Metrics",

            config = {
                'method':method,
                'GT_type': True if GTsoft else False,
                'session_type': 'single-session' if Fq.all() == Fm.all() else 'multi-session'
            }
        )

        self.Fq = Fq
        self.Fm = Fm
        self.S = np.matmul(Fq, Fm.T).T
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
        P, R = self.createPR(matching=matching)

        metrics = {"precision":prec,
                  "recall":recall,
                  "recall@1":recallAt1,
                  "recall@5":recallAt5,
                  "recall@10":recallAt10,
                  "recall@100precision":recallAt100precision,
                  "auprc":auprc}

        print(metrics)

        pr_curve = [[r, p] for (p, r) in zip(P, R)]
        table = wandb.Table(data=pr_curve, columns=["Recall", "Precision"])
        wandb.log({"Precision Recall Curve": wandb.plot.line(table, "Recall", "precision", title="Precision Recall "
                                                                                                 "Curve")})
        wandb.log(metrics)


    def precision(self, matching='single'):
        if matching == 'single':
            M = matching_methods.best_match_per_query(self.S)
        elif matching == 'auto':
            M = matching_methods.thresholding(self.S, 'auto')
        elif type(matching) == float:
            M = matching_methods.thresholding(self.S, matching)
        return sklearn.metrics.precision_score(self.GT.flatten(), M.flatten())

    def recall(self, matching='single'):
        if matching == 'single':
            M = matching_methods.best_match_per_query(self.S)
        elif matching == 'auto':
            M = matching_methods.thresholding(self.S, 'auto')
        elif type(matching) == float:
            M = matching_methods.thresholding(self.S, matching)
        return sklearn.metrics.recall_score(self.GT.flatten(), M.flatten())

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
            GTP = np.count_nonzero(GT)  # ground truth positives

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
