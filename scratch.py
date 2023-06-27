from vpr.vpr_techniques import mixvpr, hog, switchCNNprec, switchCNNf1
from vpr.data.datasets import SPED_V2
from vpr.evaluate.metrics import Metrics
import config

method = switchCNNf1

Q = SPED_V2.get_query_paths()
M = SPED_V2.get_map_paths()
GT = SPED_V2.get_gtmatrix()

S = method.matching_method(method.compute_query_desc(Q), method.compute_map_features(M))

print(S.shape)
print(GT.shape)
print(len(Q))
print(len(M))
