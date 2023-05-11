import os
import glob
import numpy as np

def get_query_paths(session_type='ms', rootdir='/home/ollie/Documents/Github/Visual-Place-Recognition'):
    if session_type=='ms':
        path = rootdir + '/data/raw_data/GardensPointWalking'
        query_paths = glob.glob(path + "/day_left/*")
        return query_paths
    else:
        raise Exception("Only session_type=ms available for this GardensPoint Dataset")



def get_map_paths(session_type='ms', rootdir='/home/ollie/Documents/Github/Visual-Place-Recognition'):
    if session_type=='ms':
        path = rootdir + '/data/raw_data/GardensPointWalking'
        test_paths = glob.glob(path + "/night_right/*")
        test_paths += glob.glob(path + "/day_right/*")
        return test_paths
    else:
        raise Exception("Only session_type=ms available for this GardensPoint Dataset")



def get_gtmatrix(session_type='ms', gt_type='hard', rootdir='/home/ollie/Documents/Github/Visual-Place-Recognition'):
    if session_type=='ms' and gt_type=='hard':
        query_paths = get_query_paths(session_type=session_type, rootdir=rootdir)
        map_paths = get_map_paths(session_type=session_type, rootdir=rootdir)
        gtmatrix = np.zeros((len(map_paths), len(query_paths)), dtype=np.uint8)

        path = rootdir + '/data/raw_data/GardensPointWalking'
        for i, m_path in enumerate(map_paths):
            for j, q_path in enumerate(query_paths):
                if 'night_right' in m_path:
                    m = m_path.replace(path + '/night_right/Image', '')
                else:
                    m = m_path.replace(path + '/day_right/Image', '')
                q = q_path.replace(path + '/day_left/Image', '')
                m = m.replace('.jpg', '')
                q = q.replace('.jpg', '')

                if int(m) == int(q):
                    gtmatrix[int(m), int(q)] = 1
        return gtmatrix
    else:
        raise Exception("Only session_type=ms and gt_type=hard available for this GardensPoint Dataset")
