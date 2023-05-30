import numpy as np
import sklearn.metrics as m
from typing import Callable
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import src.evaluate.matching as matching_methods
from sklearn.metrics import ConfusionMatrixDisplay
import sklearn
import wandb
from PIL import Image
from src.evaluate import view_matches


class Metrics:
    def __init__(self, method_name: str,
                 dataset_name: str,
                 Fq: np.ndarray,
                 Fm: np.ndarray,
                 GT: np.ndarray,
                 GTsoft: np.ndarray = None,
                 matching_method: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
                 rootdir: str=None,
                 q_pths=None, db_pths=None):
        """
        This class provides functionality for evaluating VPR techniques for a single dataset.
        It logs all computed evaluations to weights and biases dashboard so multiple techniques
        can be evaluated together.

        :param method_name: Name of the VPR technique
        :param dataset_name: Name of the Dataset
        :param Fq: Query descriptors
        :param Fm: Map descriptors
        :param GT: GT Matrix rows correspond to database images, columns correspond to query images
        :param GTsoft: GT Matrix with the soft evaluations. see https://arxiv.org/abs/2303.03281
        :param matching_method: A method that computes similarity between query and database descriptors Fq, Fm
        :param rootdir: absolute Root directory of project e.g. '../Visual-Place-Recognition'
        :param q_pths: List of absolute paths to the query images
        :param db_pths: List of absolute paths to the database images
        """
        # ========================== Setup wandb for logging results ================================================
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
        self.GT = GT
        self.GTsoft = GTsoft

        # Use a matching method or default to cosine similarity for vector space model
        if matching_method:
            self.S = matching_method(Fq, Fm)
        else:
            self.S = cosine_similarity(Fq, Fm)

    def log_metrics(self, matching='multi', threshold_type='single'):
        """
        This function logs all available metrics to the weights and biases dashboard.

        :param matching: determines the type of VPR session. Is either 'multi' session or 'single' session vpr. See 'https://arxiv.org/abs/2303.03281'
        :param threshold_type: determines how the predictions are
               computed from the similarity matrix. Is either 'single', 'auto' or a float e.g. '0.67'

        :return: None
        """

        # ================= Compute all the Metrics ===========================
        prec = self.precision(threshold_type=threshold_type)  # log precision
        recall = self.recall(threshold_type=threshold_type)  # log recall
        recallAt1 = self.recallAtK(K=1)  # log recall at k=1
        recallAt5 = self.recallAtK(K=5)  # log recall at k=5
        recallAt10 = self.recallAtK(K=10)  # log recall at k=10
        recallAt100precision = self.recallAt100precision(matching='multi')  # log recall at 100% precision
        auprc = self.AU_PRC(matching=matching)  # log area under PR curve
        d_dim, d_type, d_bytes = self.descriptor_size()

        # ================= Plotting Figures and Curves =======================
        #plot curves
        self.ROCcurve(matching=matching)  # ROC Curve
        self.PRcurve(matching=matching)  # PR Curve
        self.confusion_matrix(threshold_type=threshold_type)  # Confusion Matrix

        # Log visualizations of matches
        if self.q_pths is not None and self.db_pths is not None:
            self.view_matches(self.q_pths, self.db_pths, self.GT, self.S,
                              threshold_type=threshold_type, GTsoft=self.GTsoft)

        # =============== Logging Metrics ====================================
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
                   "auprc": [auprc],
                   "descriptor_dim": d_dim,
                   "descriptor_type": d_type,
                   "descriptor_nbytes": d_bytes}

        # Log the metrics into a wandb table
        metrics_table = wandb.Table(dataframe=pd.DataFrame.from_dict(metrics))
        self.run.log({"metrics": metrics_table})
        wandb.finish()

    def view_matches(self, q_pths: list[str], db_pths: list[str],
                     GT: np.ndarray[np.float32], S: np.ndarray[np.float32],
                     threshold_type: str = 'single', GTsoft: np.ndarray[np.float32] = None,
                     show: bool = False) -> None:
        '''
        This method produces a figure displaying images from a TP or FP VPR preidction. The figure is logged to
        weights and biases

        :param q_pths: List of absolute paths to the query images
        :param db_pths: List of absolute paths to the database images
        :param GT: GT Matrix rows correspond to database images, columns correspond to query images
        :param S: Similairy matirx row index corresponds to map image, column index corresponds to query image.
                  value at [row, index] is in the range 0-1 and corresponds to similiary of q_pths[row] to db_pths[column]
        :param threshold_type: determines how the predictions are
               computed from the similarity matrix. Is either 'single', 'auto' or a float e.g. '0.67'
        :param GTsoft: GT Matrix with the soft evaluations. see https://arxiv.org/abs/2303.03281
        :param show: True plot the figure locally False just logs to the wandb dashboard
        :return: None
        '''
        # Convert similarity matrix S into predictions matrix P
        if threshold_type == 'single':
            P = matching_methods.best_match_per_query(S)
        elif threshold_type == 'auto':
            P = matching_methods.thresholding(S, 'auto')
        elif type(threshold_type) == float:
            P = matching_methods.thresholding(S, threshold_type)

        TP = []
        FP = []

        # Use GTsoft as ground truth if available
        GT = GTsoft if isinstance(GTsoft, type(np.zeros(1))) else GT
        # Collect index's of TP and FP predictions
        for i in range(GT.shape[0]):
            for j in range(GT.shape[1]):
                if GT[i, j] == 0 and P[i, j] == 1:
                    FP.append([j, i])  # add false positive index's
                if GT[i, j] == 1 and P[i, j] == 1:
                    TP.append([j, i])  # add true positive index's

        # Collect the images of TP and FP predictions
        TP, FP = np.array(TP), np.array(FP)
        db_imgs = [np.array(Image.open(pth)) for pth in db_pths]
        q_imgs = [np.array(Image.open(pth)) for pth in q_pths]

        # Build the Figure
        img = view_matches.show(db_imgs, q_imgs, TP, FP, M=M, show=show)

        # log the figure to wandb dashboard
        wandb.log({'matches_' + self.dataset_name: wandb.Image(img)})

    def precision(self, threshold_type: str = 'single') -> float:
        '''
        Function computes VPR predictions using the similarity matrix with the given threshold type.
        the predictions are subequently used to compute the precision value.

        :param threshold_type: determines how the predictions are
               computed from the similarity matrix. Is either 'single', 'auto' or a float e.g. '0.67'
        :return: precision value
        '''

        # Compute the predictions from similarity matrix
        if threshold_type == 'single':
            P = matching_methods.best_match_per_query(self.S)
        elif threshold_type == 'auto':
            P = matching_methods.thresholding(self.S, 'auto')
        elif type(threshold_type) == float:
            P = matching_methods.thresholding(self.S, threshold_type)

        return sklearn.metrics.precision_score(self.GTsoft.flatten().astype(int), P.flatten().astype(int))

    def confusion_matrix(self, threshold_type: str = 'single') -> None:
        """
        Function computes VPR predictions using the similarity matrix with the given threshold type.
        the predictions are subequently used to compute the confusion matrix which is logged to wandb

        :param threshold_type: determines how the predictions are
               computed from the similarity matrix. Is either 'single', 'auto' or a float e.g. '0.67'
        :return: None
        """
        # Compute the predictions from similarity matrix
        if threshold_type == 'single':
            M = matching_methods.best_match_per_query(self.S)
        elif threshold_type == 'auto':
            M = matching_methods.thresholding(self.S, 'auto')
        elif type(threshold_type) == float:
            M = matching_methods.thresholding(self.S, threshold_type)

        y_truth = self.GTsoft if isinstance(self.GTsoft, type(np.ones(1))) else self.GThard

        # Create Figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Compute confusion matrix and plot it
        cm = ConfusionMatrixDisplay.from_predictions(y_truth.flatten().astype(int), M.flatten().astype(int),
                                                     display_labels=['0', '1'], ax=ax)
        # Lable the figure
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['No Match', 'Place Match'])
        ax.yaxis.set_ticklabels(['No Match', 'Place Match'])
        cm.plot()

        # Log the figure to wandb dashboard
        fig.savefig(self.rootdir + '/src/evaluate/tmp/plot.png')
        plot = Image.open(self.rootdir + '/src/evaluate/tmp/plot.png')
        wandb.log({'confmap_' + self.dataset_name: wandb.Image(plot)})

    def recall(self, threshold_type: str = 'single') -> float:
        """
        Function computes VPR predictions using the similarity matrix with the given threshold type.
        the predictions are subequently used to compute the recall

        :param threshold_type: determines how the predictions are
               computed from the similarity matrix. Is either 'single', 'auto' or a float e.g. '0.67'
        :return: None
        """
        # Compute the predictions from similarity matrix
        if threshold_type == 'single':
            P = matching_methods.best_match_per_query(self.S)
        elif threshold_type == 'auto':
            P = matching_methods.thresholding(self.S, 'auto')
        elif type(threshold_type) == float:
            P = matching_methods.thresholding(self.S, threshold_type)
        return sklearn.metrics.recall_score(self.GT.flatten().astype(int), P.flatten().astype(int))

    def createPR(self, matching: str = 'multi', n_thresh: int = 100) -> tuple[np.ndarray, np.ndarray]:
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

    def ROCcurve(self, matching: str = 'multi', n_thresh: int = 100) -> None:
        """
        Function computes and logs the Receiver Operator Curve and loggs it to weights and biases

        :param matching: determines the type of VPR session. Is either 'multi' session or 'single' session vpr. See 'https://arxiv.org/abs/2303.03281'
        :param n_thresh: The integer n_tresh controls the number of threshold values and should be >1.
        :return: None
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
        scores = np.concatenate((1 - S.flatten()[:, None], S.flatten()[:, None]), axis=1)
        wandb.log({"roc_" + self.dataset_name: wandb.plot.roc_curve(GT.flatten().astype(int),
                                                                    scores, labels=["No_Match", "Place_Match"],
                                                                    classes_to_plot=[1],
                                                                    title="ROC Curve - Dataset: " + self.dataset_name)})

    def PRcurve(self, matching: str = 'multi', n_thresh: int = 100) -> None:
        """
        Function computes the Precision Recall Curve and loggs it to weights and baises dashboard

        :param matching: determines the type of VPR session. Is either 'multi' session or 'single' session vpr. See 'https://arxiv.org/abs/2303.03281'
        :param n_thresh: The integer n_tresh controls the number of threshold values and should be >1.
        :return: None
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
        scores = np.concatenate((1 - S.flatten()[:, None], S.flatten()[:, None]), axis=1)
        wandb.log({"pr_" + self.dataset_name: wandb.plot.pr_curve(GT.flatten().astype(int), scores,
                                                                  labels=["No_Match", "Place_Match"],
                                                                  classes_to_plot=[1],
                                                                  title="PR Curve - Dataset: " + self.dataset_name)})

    def recallAtK(self, K: int = 1) -> float:
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

    def recallAt100precision(self, matching: str = 'multi', n_thresh: int = 100) -> float:
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

    def AU_PRC(self, matching: str = 'single', n_thresh: int = 100) -> float:
        """
        Computes the Area Under the Precision Recall Curve and returns it.

        :param matching: determines the type of VPR session. Is either 'multi' session or 'single' session vpr. See 'https://arxiv.org/abs/2303.03281'
        :param n_thresh: The integer n_tresh controls the number of threshold values and should be >1.
        :return: None
        """
        P, R = self.createPR(matching=matching, n_thresh=n_thresh)
        return np.trapz(P, R)

    def descriptor_size(self) -> tuple[int, type, int]:
        """
        Computes the dimensions, type and memory size of a single descriptor for the VPR method

        :return: tuple of (descriptor dimension, descriptor dtype, descriptor memory size nbytes)
        """
        type = str(self.Fm.dtype)
        nbytes = self.Fm[0].nbytes
        size = self.Fm.shape[1]
        return size, type, nbytes
