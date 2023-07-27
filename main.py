from vpr.data.datasets import GardensPointWalking, SFU, StLucia, ESSEX3IN1, Nordlands, SPED_V2, berlin_kudamm, Nordlands_passes, combined_dataset
from vpr.evaluate.metrics_wandb import Metrics
from vpr.evaluate import metrics
from vpr.evaluate.timer import Timer
from vpr.vpr_techniques.utils import load_descriptors
import config
import argparse
import importlib


parser = argparse.ArgumentParser()

parser.add_argument('--mode', required=True, choices=("describe", "eval_time", "eval_metrics_wb", "eval_metrics", "eval_invariance"),
                    help='Specify either describe or evaluate', type=str)
parser.add_argument('--datasets', choices=("SFU", "GardensPointWalking", "StLucia", "combined_dataset", "ESSEX3IN1", "Nordlands", "SPED_V2", "ESSEX3IN1", "Pittsburgh30k", "berlin_kudamm", "Nordlands_passes"),
                    help='specify one of the datasets from vpr/data/raw_data', type=str, default="StLucia", nargs='+')
parser.add_argument('--methods', choices=("patchnetvlad", "hog", "mixvpr", "alexnet", "netvlad", "cosplace", "conv_ap", "delf", "cohog", "selectCNN", "selectCNN_auprc", "selectCNN_f1", "selectCNN_fbeta", "selectCNN_precision", "selectCNN_recall", "selectCNN_recallat1", "selectCNN_recallat5", "selectCNN_recallat10"),
                    help="specify one of the techniques from vpr/vpr_tecniques", type=str, default="hog", nargs='+')
args = parser.parse_args()

# ============== Chose the datasets ===============
all_datasets = [SFU, GardensPointWalking, StLucia, ESSEX3IN1, Nordlands, SPED_V2, berlin_kudamm, berlin_kudamm, Nordlands_passes, combined_dataset]
all_datasets_name = [ds.NAME for ds in all_datasets]

datasets = []
for ds_name in args.datasets:
    datasets.append(all_datasets[all_datasets_name.index(ds_name)])

# ============= Chose the methods ================

methods_names = args.methods

# =============== Describe Mode ==================
if args.mode == "describe":
    for dataset in datasets:
        for method_name in methods_names:
            method = importlib.import_module("vpr.vpr_techniques." + method_name)
            M = dataset.get_map_paths()
            Q = dataset.get_query_paths()
            GT = dataset.get_gtmatrix(gt_type='hard')
            GTsoft = dataset.get_gtmatrix(gt_type='soft')

            method.compute_query_desc(Q, dataset_name=dataset.NAME)
            method.compute_map_features(M, dataset_name=dataset.NAME)
            del method

# ============== Evaluate Mode =======================
if args.mode == "eval_metrics_wb":
    for dataset in datasets:
        for method_name in methods_names:
            method = importlib.import_module("vpr.vpr_techniques." + method_name)
            M = dataset.get_map_paths()
            Q = dataset.get_query_paths()
            GT = dataset.get_gtmatrix(gt_type='hard')
            GTsoft = dataset.get_gtmatrix(gt_type='soft')

            Fq, Fm = load_descriptors(dataset.NAME, method.NAME)
            eval = Metrics(method.NAME, dataset.NAME, Fq, Fm, GT, matching_method=method.matching_method, q_pths=Q, db_pths=M, GTsoft=GTsoft, rootdir=config.root_dir)
            eval.run_evaluation()
            del method

if args.mode == "eval_metrics":
    for dataset in datasets:
        for dataset in datasets:
            S_data ={}
            for method_name in methods_names:
                method = importlib.import_module("vpr.vpr_techniques." + method_name)
                M = dataset.get_map_paths()
                Q = dataset.get_query_paths()
                GT = dataset.get_gtmatrix(gt_type='hard')
                GTsoft = dataset.get_gtmatrix(gt_type='soft')
                Fq, Fm = load_descriptors(dataset.NAME, method.NAME)
                S = method.matching_method(Fq, Fm)
                S_data[method.NAME] = S
            metrics.plot_curvepr(GT, S_data, dataset_name=dataset.NAME, show=False)
            metrics.plot_recallat1(GT, S_data, dataset_name=dataset.NAME, show=False)
            metrics.plot_average_precision(GT, S_data, dataset_name=dataset.NAME, show=False)
            metrics.plot_precision(GT, S_data, dataset_name=dataset.NAME, show=False)
            #metrics.compute_metrics(GT, S_data, dataset_name=dataset.NAME)
            del method

if args.mode == "eval_time":
    eval = Timer(dataset, method)
    eval.run_evaluation()

if args.mode == "eval_invariance":
    raise NotImplementedError
