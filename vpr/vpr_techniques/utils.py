import numpy as np
import os


def save_descriptors(dataset_name, method_name, desc, type='query'):
    if dataset_name is not None:
        pth = os.getcwd().replace('vpr_techniques', '') + '/vpr/descriptors/' + dataset_name
        file_name = '/m_desc.npy' if type == 'map' else '/q_desc.npy'
        if os.path.exists(pth):
            if os.path.exists(pth + '/' + method_name):
                np.save(pth + '/' + method_name + file_name, desc)
            else:
                os.mkdir(pth + '/' + method_name)
                np.save(pth + '/' + method_name + file_name, desc)
        else:
            os.mkdir(pth)
            os.mkdir(pth + '/' + method_name)
            np.save(pth + '/' + method_name + file_name, desc)


def load_descriptors(dataset_name, method_name):
    try:
        q_pth = os.getcwd().replace('vpr_technqiues', '') + '/vpr/descriptors/' + \
                dataset_name + '/' + method_name + '/q_desc.npy'
        m_pth = os.getcwd().replace('vpr_technqiues', '') + '/vpr/descriptors/' + \
                dataset_name + '/' + method_name + '/m_desc.npy'

        q_desc = np.load(q_pth)
        m_desc = np.load(m_pth)
        return q_desc, m_desc
    except:
        raise Exception("Descriptors with dataset: " + dataset_name + " and method "
                        + method_name + " are not computed. Please compute them first")
