from data.datasets import GardensPointWalking
from vpr_techniques.python import netvlad


M = GardensPointWalking.get_map_paths()
Q = GardensPointWalking.get_query_paths()
GT = GardensPointWalking.get_gtmatrix(gt_type='hard')
GTsoft = GardensPointWalking.get_gtmatrix(gt_type='soft')

F = netvlad.compute_map_features(M)