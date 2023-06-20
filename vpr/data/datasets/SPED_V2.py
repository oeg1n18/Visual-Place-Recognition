import glob
import os
import urllib
import urllib.request
import zipfile
import config
import pandas as pd
import csv

import numpy as np

NAME = 'SPED_V2'

def get_query_paths(session_type='ms'):
    return sorted(glob.glob(config.root_dir + '/vpr/data/raw_data/SPED_V2/target/*.jpg'))


def get_map_paths(session_type='ms'):
    return sorted(glob.glob(config.root_dir + '/vpr/data/raw_data/SPED_V2/ref/*.jpg'))


def get_gtmatrix(session_type='ms', gt_type='hard'):
    Q, M = get_query_paths(), get_map_paths()
    GT = np.zeros((len(Q), len(M)))
    with open(config.root_dir + '/vpr/data/raw_data/SPED_V2/gt.csv', 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            record = row[0].split(sep=';')
            if record[0] == 'N':
                if record[2] == '213_0':
                    continue
                q_idx = Q.index(config.root_dir + '/vpr/data/raw_data/SPED_V2/target/' + record[2] + '.jpg')
                for reference in record[3:]:
                    m_idx = M.index(config.root_dir + '/vpr/data/raw_data/SPED_V2/ref/' + reference + '.jpg')
                    GT[q_idx, m_idx] = 1
        return GT.astype(np.uint8)