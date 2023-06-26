#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import faiss
"""
Created on Thu Mar 26 14:49:42 2020

@author: mubariz
"""
import cv2
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from vpr.vpr_techniques.utils import save_descriptors
from tqdm import tqdm

NAME = 'HOG'

def compute_query_desc(Q, dataset_name=None, disable_pbar=False):
    ref_map = [cv2.imread(pth, 0) for pth in Q]

    winSize = (512, 512)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (16, 16)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    ref_desc_list = []
    for ref_image in tqdm(ref_map, desc='Computing Query Descriptors', disable=disable_pbar):
        if ref_image is not None:
            hog_desc = hog.compute(cv2.resize(ref_image, winSize))
        ref_desc_list.append(hog_desc)
    q_desc = np.array(ref_desc_list).astype(np.float32)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, q_desc, type='query')
    return q_desc

def compute_map_features(M, dataset_name=None, disable_pbar=False):
    ref_map = [cv2.imread(pth, 0) for pth in M]

    winSize = (512, 512)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (16, 16)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    ref_desc_list = []
    for ref_image in tqdm(ref_map, desc='Computing Map Descriptors', disable=disable_pbar):
        if ref_image is not None:
            hog_desc = hog.compute(cv2.resize(ref_image, winSize))
        ref_desc_list.append(hog_desc)
    m_desc = np.array(ref_desc_list).astype(np.float32)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, m_desc, type='map')
    return m_desc


def matching_method(q_desc, m_desc):
    return cosine_similarity(q_desc, m_desc).transpose()

class PlaceRecognition:
    def __init__(self, m_desc):
        self.m_desc = m_desc
        self.index = faiss.IndexFlatL2(m_desc.shape[1])
        self.index.add(m_desc)

    def perform_vpr(self, q_path):
        q_desc = compute_query_desc(q_path, disable_pbar=True)
        D, I = self.index.search(q_desc.astype(np.float32), 1)
        temp_mdesc = self.m_desc[I].squeeze() if self.m_desc[I].squeeze().ndim == 2 else self.m_desc[I][0]
        scores = cosine_similarity(q_desc, temp_mdesc).diagonal()
        return I.flatten(), scores







