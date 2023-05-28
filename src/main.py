import argparse
from src.data.datasets import GardensPointWalking
from src.data.datasets import StLucia
from src.data.datasets import SFU
from src.vpr_techniques.python import densevlad
#from src.vpr_techniques.python import patchnetvlad
#from src.vpr_techniques.python import netvlad
from evaluate.metrics import Metrics
import config

"""
parser = argparse.ArgumentParser()


parser.add_argument('--mode', required=True, choices=("describe", "evaluate", "visualise"), help='Specify either describe or evaluate', type=str)
parser.add_argument('--dataset', choices=("Nordlands", "Pittsburgh", "ESSEX3IN1"), help='specify one of the datasets from src/data/raw_data', type=str)
parser.add_argument('--technique', choices=("patchnetvlad", "densevlad"), help="specify one of the techniques from src/vpr_tecniques", type=str)
parser.add_argument('--language', choices=("python, cpp"), help="specify either python or cpp", type=str)

args = parser.parse_args()
"""

dataset = GardensPointWalking
dataset.download(config.root_dir)

M = dataset.get_map_paths(rootdir=config.root_dir)
Q = dataset.get_query_paths(rootdir=config.root_dir)
GT = dataset.get_gtmatrix(rootdir=config.root_dir, gt_type='hard')
GTsoft = dataset.get_gtmatrix(rootdir=config.root_dir, gt_type='soft')

Fq = densevlad.compute_query_desc(Q, dataset_name=dataset.NAME)
Fm = densevlad.compute_map_features(M, dataset_name=dataset.NAME)

eval = Metrics(densevlad.NAME, dataset.NAME, Fq, Fm, GT, GTsoft=GTsoft, rootdir=config.root_dir)
eval.log_metrics()

"""Fq = netvlad.compute_query_desc(Q)
Fm = netvlad.compute_map_features(M)

eval = Metrics(netvlad.NAME, dataset.NAME, Fq, Fm, GT, GTsoft=GTsoft, rootdir=config.root_dir)
eval.log_metrics()

Fq = patchnetvlad.compute_query_desc(Q)
Fm = patchnetvlad.compute_map_features(M)

eval = Metrics(patchnetvlad.NAME, dataset.NAME, Fq, Fm, GT, GTsoft=GTsoft,
               matching_method=patchnetvlad.matching_function, rootdir=config.root_dir)
eval.log_metrics()"""
