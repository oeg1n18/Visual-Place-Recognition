import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

class Vis:
    def __init__(self, Fq, Fm, GT):
        self.Fm = Fm
        self.Fq = Fq
        self.S = np.matmul(Fq, Fm.T)

    def dashboard_single(self, metrics):
        fig, ax = plt.subplots(2,2)
        ax[0][0].imshow(self.S)
        ax[0][0].get_xaxis().set_visible(False)
        ax[0][0].get_yaxis().set_visible(False)
        ax[0][0].set_title("Similarity Matrix")

        ax[0][1].plot(metrics["PR_recall"], metrics["PR_precision"])
        ax[0][1].set_xlabel("Recall")
        ax[0][1].set_ylabel("Precision")
        ax[0][1].set_title("PR-Curve")

        labels = ["recall@100precision", "recall@1", "recall@5", "recall@10"]
        ax[1][0].bar(labels, [metrics["recall@100precision"], metrics["recall@1"], metrics["recall@5"], metrics["recall@10"]])
        ax[1][0].set_xticks([1, 2, 3, 4])
        ax[1][0].set_xticklabels(labels, rotation=-30)
        ax[1][0].set_title("Recall Metrics")


        labels = ["precision", "recall", "auprc"]
        ax[1][1].bar(labels, [metrics["precision"], metrics["recall"], metrics["auprc"]])

        plt.xticks(rotation=-30)
        plt.tight_layout()
        plt.show()


