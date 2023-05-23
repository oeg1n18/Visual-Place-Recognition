import argparse
from src.data.datasets import GardensPointWalking
from src.vpr_techniques.python import densevlad
from src.vpr_techniques.python import patchnetvlad
from src.vpr_techniques.python import netvlad
from evaluate.metrics import Metrics
import os

PROJECT_ROOT = os.getcwd().replace('/main.py', '')

"""
parser = argparse.ArgumentParser()


parser.add_argument('--mode', required=True, choices=("describe", "evaluate", "visualise"), help='Specify either describe or evaluate', type=str)
parser.add_argument('--dataset', choices=("Nordlands", "Pittsburgh", "ESSEX3IN1"), help='specify one of the datasets from src/data/raw_data', type=str)
parser.add_argument('--technique', choices=("patchnetvlad", "densevlad"), help="specify one of the techniques from src/vpr_tecniques", type=str)
parser.add_argument('--language', choices=("python, cpp"), help="specify either python or cpp", type=str)

args = parser.parse_args()
"""

M = GardensPointWalking.get_map_paths(rootdir=PROJECT_ROOT)
Q = GardensPointWalking.get_query_paths(rootdir=PROJECT_ROOT)
GT = GardensPointWalking.get_gtmatrix(rootdir=PROJECT_ROOT, gt_type='hard')
GTsoft = GardensPointWalking.get_gtmatrix(rootdir=PROJECT_ROOT, gt_type='soft')



Fq = densevlad.compute_query_desc(Q)
Fm = densevlad.compute_map_features(M)

eval = Metrics(densevlad.NAME, GardensPointWalking.NAME, Fq, Fm, GT, GTsoft=GTsoft)
eval.log_metrics()

Fq = netvlad.compute_query_desc(Q)
Fm = netvlad.compute_map_features(M)

eval = Metrics(netvlad.NAME, GardensPointWalking.NAME, Fq, Fm, GT, GTsoft=GTsoft)
eval.log_metrics()

Fq = patchnetvlad.compute_query_desc(Q)
Fm = patchnetvlad.compute_map_features(M)

eval = Metrics(patchnetvlad.NAME, GardensPointWalking.NAME, Fq, Fm, GT, GTsoft=GTsoft, matching_method=patchnetvlad.matching_function)
eval.log_metrics()

