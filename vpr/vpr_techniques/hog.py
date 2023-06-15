#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:49:42 2020

@author: mubariz
"""
import cv2
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from vpr.vpr_techniques.utils import save_descriptors

NAME = 'HOG'

def compute_query_desc(Q, dataset_name=None):
    ref_map = [cv2.imread(pth, 0) for pth in Q]

    winSize = (512, 512)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (16, 16)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    ref_desc_list = []
    for ref_image in ref_map:
        if ref_image is not None:
            hog_desc = hog.compute(cv2.resize(ref_image, winSize))
        ref_desc_list.append(hog_desc)
    q_desc = np.array(ref_desc_list).astype(np.float32)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, q_desc, type='query')
    return q_desc

def compute_map_features(M, dataset_name=None):
    ref_map = [cv2.imread(pth, 0) for pth in M]

    winSize = (512, 512)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (16, 16)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    ref_desc_list = []
    for ref_image in ref_map:
        if ref_image is not None:
            hog_desc = hog.compute(cv2.resize(ref_image, winSize))
        ref_desc_list.append(hog_desc)
    m_desc = np.array(ref_desc_list).astype(np.float32)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, m_desc, type='map')
    return m_desc


def perform_vpr(q_path, m_desc):
    q_desc = compute_query_desc([q_path])
    S = matching_method(q_desc, m_desc)
    i, j = np.unravel_index(S.argmax(), S.shape)
    return int(j), float(S[i,j])


def matching_method(q_desc, m_desc):
    return cosine_similarity(q_desc, m_desc)


