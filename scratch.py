from vpr.vpr_techniques import mixvpr
from vpr.data.datasets import SPED_V2
from vpr.evaluate.metrics import Metrics
import config

ds = SPED_V2
method = mixvpr
Q = ds.get_query_paths()
M = ds.get_map_paths()
GT = ds.get_gtmatrix()

q_desc = method.compute_query_desc([Q[0]])
m_desc = method.compute_map_features(M)

S = method.matching_method(q_desc, m_desc)
#eval = Metrics(method.NAME, ds.NAME, q_desc, m_desc, GT, matching_method=method.matching_method, q_pths=Q, db_pths=M,
print(S.shape)