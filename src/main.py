import argparse
from data.datasets import GardensPointWalking
from vpr_techniques.python import densevlad
from evaluate.metrics import Metrics

"""
parser = argparse.ArgumentParser()


parser.add_argument('--mode', required=True, choices=("describe", "evaluate", "visualise"), help='Specify either describe or evaluate', type=str)
parser.add_argument('--dataset', choices=("Nordlands", "Pittsburgh", "ESSEX3IN1"), help='specify one of the datasets from src/data/raw_data', type=str)
parser.add_argument('--technique', choices=("patchnetvlad", "densevlad"), help="specify one of the techniques from src/vpr_tecniques", type=str)
parser.add_argument('--language', choices=("python, cpp"), help="specify either python or cpp", type=str)

args = parser.parse_args()
"""


M = GardensPointWalking.get_map_paths()
Q = GardensPointWalking.get_query_paths()
GT = GardensPointWalking.get_gtmatrix(gt_type='hard')
GTsoft = GardensPointWalking.get_gtmatrix(gt_type='soft')

print(GT.shape)
print(GTsoft.shape)

Fq = densevlad.compute_query_desc(Q)
Fm = densevlad.compute_map_features(M)



eval = Metrics('DenseVLAD', Fq, Fm, GT, GTsoft=GTsoft)
eval.log_metrics()