import numpy as np
import sklearn.metrics as m
import matplotlib.pyplot as plt
import time


class Evaluate:
    def __init__(self, vpr, dm, device='cpu', batch_size=32):
        self.vpr = vpr
        self.dm = dm
        self.GT = dm.GT
        self.M = vpr.compute_map_desc(dm.M)
        self.Q = vpr.compute_query_desc(dm.Q)
        self.S = vpr.similarity_matrix(dm.Q)

    def profile(self, threshold=0.7, N=10, p=1., k=1000):
        print("Precision: ", self.precision(threshold=threshold))
        print("Recall: ", self.recall(threshold=threshold))
        print("f1: ", self.f1())
        print("AP: ", self.average_precision())
        print("recall@" + str(p), "p: ", self.recall_at_p(p=p))
        print("recall@" + str(k), "k: ", self.recall_at_k(k=k))
        t, v = self.encoding_time(N)
        print("Encoding Time: avg=", f'{t:.4}', "s var=", f'{v:.4}', "device=", str(self.vpr.device))
        t, v = self.retrieval_time(N)
        print("Retrieval Time: avg=", f'{t:.4}', "s var=", f'{v:.4}', "device=", str(self.vpr.device))
        self.descriptor_size()

    def precision(self, threshold=0.7):
        y_true = self.GT.flatten()
        s_flat = self.S.flatten()
        y_pred = (s_flat > threshold) * np.ones_like(s_flat)
        y_pred = y_pred.astype(np.uint8)
        return m.precision_score(y_true, y_pred)

    def recall(self, threshold=0.7):
        y_true = self.GT.flatten()
        s_flat = self.S.flatten()
        y_pred = (s_flat > threshold) * np.ones_like(s_flat)
        y_pred = y_pred.astype(np.uint8)
        return m.recall_score(y_true, y_pred)

    def f1(self, threshold=0.7):
        y_true = self.GT.flatten()
        s_flat = self.S.flatten()
        y_pred = (s_flat > threshold) * np.ones_like(s_flat)
        y_pred = y_pred.astype(np.uint8)
        return m.f1_score(y_true, y_pred)

    def precision_recall_curve(self, show=False):
        y_true = self.GT.flatten()
        y_scores = self.S.flatten()
        if show:
            m.PrecisionRecallDisplay.from_predictions(y_true, y_scores)
            plt.show()
        else:
            precision, recall, thresholds = m.precision_recall_curve(y_true, y_scores)
            return precision, recall, thresholds

    def average_precision(self):
        y_true = self.GT.flatten()
        y_scores = self.S.flatten()
        return m.average_precision_score(y_true, y_scores)

    def recall_at_p(self, p):
        precision, recall, thresholds = self.precision_recall_curve(show=False)
        for i, prec in enumerate(np.flip(precision)):
            if prec < p:
                return np.flip(recall)[i - 1]

    def recall_at_k(self, k, threshold=0.5):
        y_true = self.GT.flatten()
        y_scores = self.S.flatten()
        i_sorted = np.flip(np.argsort(y_scores))
        y_true = y_true[i_sorted]
        y_scores = y_scores[i_sorted]
        y_pred = (y_scores > threshold) * np.ones_like(y_scores)
        y_pred = y_pred.astype(np.uint8)
        return m.recall_score(y_true[:k], y_pred[:k])

    def viewpoint_inv(self):
        raise NotImplementedError()

    def illumination_inv(self):
        raise NotImplementedError()

    def descriptor_size(self):
        type = str(self.M.dtype)
        nbytes = self.M[0].nbytes
        size = self.M.shape[1]
        print("Descriptor has size: ", size, " of type ", type, "taking up ", nbytes, " bytes")

    def retrieval_time(self, N=10):
        times = []
        for q in self.dm.Q[:N]:
            start_time = time.time()
            P, S = self.vpr.perform_vpr(q)
            end_time = time.time()
            times.append(end_time - start_time)
        return np.mean(times), np.var(times)

    def encoding_time(self, N=10):
        times = []
        for q in self.dm.Q[:N]:
            start_time = time.time()
            desc = self.vpr.compute_query_desc([q])
            end_time = time.time()
            times.append(end_time - start_time)
        return np.mean(times), np.var(times)
