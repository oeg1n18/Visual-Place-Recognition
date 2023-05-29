import argparse
from src.data.datasets import GardensPointWalking
from src.data.datasets import StLucia
from src.data.datasets import SFU
from src.vpr_techniques.python import densevlad
from src.vpr_techniques.python import patchnetvlad
from src.vpr_techniques.python import netvlad
from src.vpr_techniques.python import mixvpr
from evaluate.metrics import Metrics
import config

parser = argparse.ArgumentParser()

parser.add_argument('--mode', required=True, choices=("describe", "eval_time", "eval_metrics", "eval_invariance"),
                    help='Specify either describe or evaluate', type=str)
parser.add_argument('--dataset', choices=("SFU", "GardensPointWalking", "StLucia"),
                    help='specify one of the datasets from src/data/raw_data', type=str, default="StLucia")
parser.add_argument('--technique', choices=("patchNetVLAD", "denseVLAD", "MixVPR", "NetVLAD"),
                    help="specify one of the techniques from src/vpr_tecniques", type=str, default="patchNetVLAD")
args = parser.parse_args()

# ============== Chose the dataset ===============
if args.dataset == "StLucia":
    dataset = StLucia
elif args.dataset == "GardensPointWalking":
    dataset = GardensPointWalking
elif args.dataset == "SFU":
    dataset = SFU
else:
    dataset = GardensPointWalking

# ============= Chose the method ================
if args.technique == "patchNetVLAD":
    method = patchnetvlad
elif args.technique == "denseVLAD":
    method = densevlad
elif args.technique == "NetVLAD":
    method = netvlad
elif args.technqiue == "MixVPR":
    method = mixvpr
else:
    method = patchnetvlad


# =============== Describe Mode ==================
if args.mode == "describe":
    M = dataset.get_map_paths(rootdir=config.root_dir)
    Q = dataset.get_query_paths(rootdir=config.root_dir)
    GT = dataset.get_gtmatrix(rootdir=config.root_dir, gt_type='hard')
    GTsoft = dataset.get_gtmatrix(rootdir=config.root_dir, gt_type='soft')

    method.compute_query_desc(Q, dataset_name=dataset.NAME)
    method.compute_map_features(M, dataset_name=dataset.NAME)


# ============== Evaluate Mode =======================


dataset = GardensPointWalking
M = dataset.get_map_paths(rootdir=config.root_dir)
Q = dataset.get_query_paths(rootdir=config.root_dir)
GT = dataset.get_gtmatrix(rootdir=config.root_dir, gt_type='hard')
GTsoft = dataset.get_gtmatrix(rootdir=config.root_dir, gt_type='soft')
"""
Fq = densevlad.compute_query_desc(Q, dataset_name=dataset.NAME)
Fm = densevlad.compute_map_features(M, dataset_name=dataset.NAME)

eval = Metrics(densevlad.NAME, dataset.NAME, Fq, Fm, GT, q_pths=Q, db_pths=M, GTsoft=GTsoft, rootdir=config.root_dir)
eval.log_metrics()
"""
Fq = mixvpr.compute_query_desc(Q)
Fm = mixvpr.compute_map_features(M)

eval = Metrics(mixvpr.NAME, dataset.NAME, Fq, Fm, GT, q_pths=Q, db_pths=M, GTsoft=GTsoft, rootdir=config.root_dir)
eval.log_metrics()

Fq = patchnetvlad.compute_query_desc(Q)
Fm = patchnetvlad.compute_map_features(M)

eval = Metrics(patchnetvlad.NAME, dataset.NAME, Fq, Fm, GT, q_pths=Q, db_pths=M, GTsoft=GTsoft,
               matching_method=patchnetvlad.matching_function, rootdir=config.root_dir)
eval.log_metrics()
