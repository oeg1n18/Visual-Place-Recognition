from vpr.data.datasets import pittsburgh30k
from vpr.evaluate.metrics_wb import Metrics
from vpr.data.utils import view_dataset_matches
import config
from vpr.data.datasets import SFU, GardensPointWalking
from vpr.vpr_techniques import patchnetvlad
from vpr.vpr_techniques.utils import load_descriptors
from vpr.data.datasets import Nordlands
from vpr.vpr_techniques import alexnet

ds = Nordlands
method = alexnet

Q = ds.get_query_paths()
M = ds.get_map_paths()

q_desc = method.compute_query_desc(Q, dataset_name=ds.NAME)
m_desc = method.compute_map_features(M, dataset_name=ds.NAME)
q_desc, m_desc = load_descriptors(ds.NAME, method.NAME)
print(q_desc.shape, m_desc.shape)
print(q_desc.shape, m_desc.shape, type(q_desc), type(m_desc))

S = method.matching_method(q_desc, m_desc)
print(S.shape)