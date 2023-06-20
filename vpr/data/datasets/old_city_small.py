import glob
import os
import urllib
import urllib.request
import zipfile
import config
import picle
import numpy as np



def get_query_paths(session_type='ms'):
    return sorted(glob(config.root_dir + '/vpr/data/raw_data/old_city_200/old_city_2/*.png))



def get_map_paths():
    return sorted(glob(config.root_dir + '/vpr/data/raw_data/old_city_200/old_city_2/*.png'))


def get_gt_matrix(session_type='ms'):
    with open(config.root_dir + '/vpr/data/raw_data/old_city_200/gtQ