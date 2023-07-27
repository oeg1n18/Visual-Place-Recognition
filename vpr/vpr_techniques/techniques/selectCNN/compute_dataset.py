from typing import Union, Any

from sklearn.metrics import precision_score, recall_score, precision_recall_curve, f1_score, average_precision_score, fbeta_score

from vpr.vpr_techniques import cosplace, mixvpr, netvlad, conv_ap
from vpr.data.datasets import Nordlands_passes, SFU
import config
import numpy as np
from vpr.evaluate.matching import thresholding
import pandas as pd
from tqdm import tqdm

def recall_at_n(all_gt: np.ndarray, all_probs: np.ndarray, all_preds: np.ndarray, n: int=5) -> float:
    recall_at_ks = []
    for i, probs in enumerate(all_probs):
        preds = all_preds[i]
        gt = all_gt[i]
        recalled = 0
        rel = 0
        top_probs_idx = np.flip(np.argsort(probs))
        for j, idx in enumerate(top_probs_idx[:n]):
            if gt[idx] == 1 and preds[idx] == 1:
                recalled += 1
                rel += 1
            elif gt[idx] == 1 and preds[idx] == 0:
                rel += 1
        if rel == 0:
            recall_at_ks.append(0)
        else:
            recall_at_ks.append(recalled/rel)
    return np.mean(recall_at_ks)


def compute_metric(gt: np.ndarray, preds: np.ndarray, probs: np.ndarray, metric: str):

    if metric == "precision":
        return precision_score(gt.flatten(), preds.flatten(), zero_division=0.0)

    elif metric == "f1_score":
        return f1_score(gt.flatten(), preds.flatten(), zero_division=0.0)

    elif metric == "recall":
        return recall_score(gt.flatten(), preds.flatten(), zero_division=0.0)

    elif metric == "recall@1":
        return recall_at_n(gt, probs, preds, 1)

    elif metric == "recall@3":
        return recall_at_n(gt, probs, preds, 3)

    elif metric == "recall@5":
        return recall_at_n(gt, probs, preds, 5)

    elif metric == "recall@10":
        return recall_at_n(gt, probs, preds, 10)

    elif metric == 'auprc':
        return average_precision_score(gt.flatten(), probs.flatten())

    elif metric == 'F-beta':
        return fbeta_score(gt.flatten(), preds.flatten(), beta=1.8)
    else:
        raise Exception("Metric " + metric + "not implemented")


def compute_data(technique, datasets,
                 metrics=["precision", "f1_score", "accuracy", "recall@3", "recall@5", "recall@10"], partition='train', augmentations=None):
    all_images = []
    all_metrics = [[] for _ in range(len(metrics))]
    for dataset in datasets:
        print(' ')
        print('================= Dataset: ', dataset.NAME, ' Technique: ', technique.NAME, '=================')
        M = dataset.get_map_paths(partition=partition)
        Q = dataset.get_query_paths(partition=partition)
        GT = dataset.get_gtmatrix(partition=partition)
        query_descriptors = technique.compute_query_desc(Q)
        map_descriptors = technique.compute_map_features(M)
        S0 = technique.matching_method(query_descriptors, map_descriptors)
        all_images = Q
        """ Speed this up it is incredibly slow """
        for i in tqdm(range(len(Q)), desc="Computing Metrics", total=len(Q)):
            if i == 0:
                S = np.array([S0[:, 0].flatten()])
                gt = np.array([GT[:, 0].flatten().astype(int)])
            elif 0 < i <= 5:
                S = S0[:, :i].transpose()
                gt = GT[:, :i].transpose()
            elif i == S0.shape[1]:
                S = np.array([S0[:, i].flatten()])
                gt = np.array([GT[:, i].flatten().astype(int)])
            else:
                S = S0[:, i-5:i].transpose()
                gt = GT[:, i-5:i].transpose()
            preds = thresholding(S, thresh='auto')
            for i, metric in enumerate(metrics):
                all_metrics[i].append(compute_metric(gt, preds, S, metric))
    return all_images, all_metrics


def build_datasets(techniques, datasets,
                   metrics=["precision", "f1_score", "accuracy", "recall@3", "recall@5", "recall@10"],
                   partition='train'):
    all_data = [{} for _ in range(len(metrics))]
    flag = 1
    for technique in techniques:
        images, all_results = compute_data(technique, datasets, metrics=metrics, partition=partition)
        if flag == 1:
            for data in all_data:
                data["query_images"] = images
            flag = 0
        for i, result in enumerate(all_results):
            all_data[i][technique.NAME] = result
    all_df = [pd.DataFrame.from_dict(data) for data in all_data]

    # save all different metrics as an individual dataset
    for i, df in enumerate(all_df):
        if len(datasets) == 1:
            df.to_csv(
                config.root_dir + '/vpr/vpr_techniques/techniques/selectCNN/data/' + datasets[0].NAME + '_' + metrics[
                    i] + '_' + partition + '.csv')
        if len(datasets) > 1:
            name = ''
            for ds in datasets:
                name += ds.NAME + '__'
            df.to_csv(config.root_dir + '/vpr/vpr_techniques/techniques/selectCNN/data/' + name + '_' + metrics[
                i] + '_' + partition + '.csv')


if __name__ == '__main__':
    techniques = [mixvpr, conv_ap, cosplace, netvlad]
    datasets = [Nordlands_passes]
    metrics = ["F-beta", "f1_score", "precision", "recall", "recall@1", "recall@3", "recall@5", "recall@10", "auprc"]
    build_datasets(techniques, datasets, metrics, partition='train')
