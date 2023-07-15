from typing import Union, Any

from sklearn.metrics import precision_score, recall_score, precision_recall_curve, f1_score, average_precision_score, fbeta_score

from vpr.vpr_techniques import cosplace, mixvpr, netvlad, hog, delf, patchnetvlad, cohog
from vpr.data.datasets import StLucia, SFU, GardensPointWalking, ESSEX3IN1, SPED_V2, berlin_kudamm, Nordlands, \
    RobotCars_short, pittsburgh30k, Nordlands_passes
import config
import numpy as np
from vpr.evaluate.matching import thresholding
import pandas as pd

def recall_at_n(gt: np.ndarray, probs: np.ndarray, n_queries: int, n: int=5) -> float:
    all_gt = np.array_split(gt, n_queries)
    all_probs = np.array_split(probs, n_queries)
    positives = 0
    for i, gt in enumerate(all_gt):
        idx = np.flip(np.argsort(all_probs[i]))
        if np.sum(gt[idx][:n] >= 1):
            positives += 1
        else:
            continue
    return positives/n_queries


def compute_metric(gt: np.ndarray, preds: np.ndarray, probs: np.ndarray, metric: str, n_prev_queries=5):
    if metric == "precision":
        return precision_score(gt, preds, zero_division=0.0)

    elif metric == "f1_score":
        return f1_score(gt, preds, zero_division=0.0)

    elif metric == "accuracy":
        if gt[np.argmax(preds)] == 1:
            return 1
        else:
            return 0

    elif metric == "recall":
        return recall_score(gt, preds, zero_division=0.0)

    elif metric == "recall@3":
        return recall_at_n(gt, probs, n_prev_queries, 3)

    elif metric == "recall@5":
        return recall_at_n(gt, probs, n_prev_queries, 5)

    elif metric == "recall@10":
        return recall_at_n(gt, probs, n_prev_queries, 10)

    elif metric == 'auprc':
        return average_precision_score(gt, probs)

    elif metric == 'F-beta':
        return fbeta_score(gt, preds, beta=1.8)
    else:
        raise Exception("Metric " + metric + "not implemented")


def compute_data(technique, datasets,
                 metrics=["precision", "f1_score", "accuracy", "recall@3", "recall@5", "recall@10"], partition='train'):
    all_images = []
    all_metrics = [[] for _ in range(len(metrics))]
    for dataset in datasets:
        print('================= Dataset: ', dataset.NAME, ' Technique: ', technique.NAME, '=================')
        M = dataset.get_map_paths(partition=partition)
        Q = dataset.get_query_paths(partition=partition)
        GT = dataset.get_gtmatrix(partition=partition)
        query_descriptors = technique.compute_query_desc(Q)
        map_descriptors = technique.compute_map_features(M)
        S0 = technique.matching_method(query_descriptors, map_descriptors)
        all_images += Q
        for i, q in enumerate(query_descriptors):
            if i == 0:
                S = S0[:, 0].flatten()
                gt = GT[:, 0].flatten().astype(int)
            elif i <= 5:
                S = S0[:, :i].flatten()
                gt = GT[:, :i].flatten().astype(int)
            else:
                S = S0[:, i-5:i].flatten()
                gt = GT[:, i-5:i].flatten().astype(int)
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
    techniques = [netvlad, hog, cosplace, mixvpr]
    datasets = [Nordlands_passes]
    metrics = ["precision", "f1_score", "accuracy", "recall@3", "recall@5", "recall@10", "auprc", "F-beta"]
    build_datasets(techniques, datasets, metrics, partition='train')
