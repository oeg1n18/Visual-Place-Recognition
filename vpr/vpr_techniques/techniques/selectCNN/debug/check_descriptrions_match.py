from vpr.vpr_techniques import netvlad, hog, cosplace, mixvpr, switchCNNprec
from vpr.data.datasets import Nordlands

techniques = [netvlad, hog, cosplace, mixvpr]


def check_dataset(Q):
    q_descs = [technique.compute_query_desc(Q) for technique in techniques]
    # m_descs = [technique.compute_map_features(M) for technique in techniques]
    q_cnn = switchCNN.compute_query_desc(Q)
    # m_cnn = selectCNN.compute_map_features(M)

    for i in range(len(Q)):
        print(q_cnn[0][i][:5], q_descs[q_cnn[1][i]][i][:5])

Q = Nordlands.get_query_paths()[:10]
check_dataset(Q)
