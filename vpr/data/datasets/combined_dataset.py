import config
from glob import glob
from vpr.data.datasets import GardensPointWalking, SFU, StLucia, Nordlands_passes
import numpy as np

NAME = 'combined_dataset'
def get_query_paths(session_type='ms', partition=None):
    Q = Nordlands_passes.get_query_paths(partition="test")
    Q += SFU.get_query_paths()
    Q += StLucia.get_query_paths()
    Q += GardensPointWalking.get_query_paths()
    return Q

def  get_map_paths(session_type='ms', partition=None):
    M = Nordlands_passes.get_map_paths(partition="test")
    M += SFU.get_map_paths()
    M += StLucia.get_map_paths()
    M += GardensPointWalking.get_map_paths()
    return M
def get_gtmatrix(session_type='ms', gt_type='hard', partition=None):
    gt_nordlands = Nordlands_passes.get_gtmatrix(gt_type=gt_type)
    gt_sfu = SFU.get_gtmatrix(gt_type=gt_type)
    gt_stlucia = StLucia.get_gtmatrix(gt_type=gt_type)
    gt_gardens = GardensPointWalking.get_gtmatrix(gt_type=gt_type)

    gts = [gt_nordlands, gt_sfu, gt_stlucia, gt_gardens]

    count = 0
    total_count = gt_nordlands.shape[1] + gt_sfu.shape[1] + gt_stlucia.shape[1] + gt_gardens.shape[1]
    gt_rows = []
    for i in range(4):
        if i == 0:
            right = np.zeros((gt_nordlands.shape[0], total_count - gt_nordlands.shape[1]))
            gt_rows.append(np.hstack((gt_nordlands, right)))
        if i == 1:
            left = np.zeros((gt_sfu.shape[0], gt_nordlands.shape[1]))
            right = np.zeros((gt_sfu.shape[0], gt_stlucia.shape[1] + gt_gardens.shape[1]))
            gt_rows.append(np.hstack((left, gt_sfu, right)))
        if i == 2:
            left = np.zeros((gt_stlucia.shape[0], gt_nordlands.shape[1] + gt_sfu.shape[1]))
            right = np.zeros((gt_stlucia.shape[0], gt_gardens.shape[1]))
            gt_rows.append(np.hstack((left, gt_stlucia, right)))
        if i == 3:
            left = np.zeros((gt_gardens.shape[0], total_count - gt_gardens.shape[1]))
            gt_rows.append(np.hstack((left, gt_gardens)))
    for row in gt_rows:
        print("=", row.shape)
    return np.vstack(gt_rows).astype(np.uint8)

def download():
    print("Seek permission of dataset owner")

