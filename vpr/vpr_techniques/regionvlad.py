from _collections import defaultdict
import config
import caffe
import pickle
import numpy as np
from PIL import Image
from vpr.vpr_techniques.techniques.regionvlad.utils import load_obj, binaryProto2npy, getVLAD, getROIs
from skimage.measure import regionprops,label
import itertools
import time
import os
from os.path import dirname

from vpr.vpr_techniques.utils import save_descriptors

# Paths to protext, model and mean file
protxt = config.root_dir + "/vpr/vpr_techniques/techniques/regionvlad/AlexnetPlaces365/deploy_alexnet_places365.prototxt"
model = config.root_dir + "/vpr/vpr_techniques/techniques/regionvlad/AlexnetPlaces365/alexnet_places365.caffemodel"
mean = config.root_dir + "/vpr/vpr_techniques/techniques/regionvlad/AlexnetPlaces365/places365CNN_mean.binaryproto"

N = 400 #No. of ROIs
layer = 'conv3'
Features, StackedFeatures = defaultdict(list),defaultdict(list)
set_gpu = False
gpu_id = 0
totalT = 0

if set_gpu:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
else:
    caffe.set_mode_cpu()
# =====================================================================

NAME = "RegionVLAD"

def matching_function(q_desc_patches, m_desc_patches):
    return None
def compute_query_desc(Q, dataset_name=None):
    net = caffe.Net(protxt, model, caffe.TEST)
    batch_size = 1
    inputSize = net.blobs['data'].shape[2]
    net.blobs['data'].reshape(batch_size, 3, inputSize, inputSize)
    ref_map_images_descs = []
    # Configuration 1
    vocab = load_obj(config.root_dir + "/vpr/vpr_techniques/techniques/regionvlad/Vocabulary/Vocabulary_100_200_300_Protocol2.pkl")
    # Configuration 2
    # vocab = load_obj(config.root_dir + "/vpr/vpr_techniques/techniques/regionvlad/Vocabulary_400_Protocol2.pkl"))

    # Set Caffe
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    mean_file = binaryProto2npy(mean)
    transformer.set_mean('data', mean_file.mean(1).mean(1))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255)
    for img_pth in Q:
        img = np.array(Image.open(img_pth))
        net.blobs['data'].data[...] = transformer.preprocess('data', img / 255.0)
        # Forward Pass
        res = net.forward()
        # Stack the activations of feature maps to make local descriptors
        Features[layer] = np.array(net.blobs[layer].data[0].copy())
        StackedFeatures[layer] = Features[layer].transpose(1, 2, 0)
        # Retrieve N ROIs for test and ref images
        ROIs = getROIs(Features[layer], StackedFeatures[layer], img)
        vocabulary = vocab[N][V][layer]
        # Retrieve VLAD descriptors using ROIs and vocabulary
        VLAD = getVLAD(ROIs, vocabulary)
        ref_map_images_descs.append(VLAD)
    q_desc = np.stack(ref_map_images_descs).astype(np.float32)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, q_desc, type='query')
    return q_desc


def compute_map_features(M, dataset_name=None):
    net = caffe.Net(protxt, model, caffe.TEST)
    batch_size = 1
    inputSize = net.blobs['data'].shape[2]
    net.blobs['data'].reshape(batch_size, 3, inputSize, inputSize)
    ref_map_images_descs = []
    # Configuration 1
    vocab = load_obj(config.root_dir + "/vpr/vpr_techniques/techniques/regionvlad/Vocabulary/Vocabulary_100_200_300_Protocol2.pkl")
    # Configuration 2
    #vocab = load_obj(config.root_dir + "/vpr/vpr_techniques/techniques/regionvlad/Vocabulary_400_Protocol2.pkl"))

    # Set Caffe
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    mean_file = binaryProto2npy(mean)
    transformer.set_mean('data', mean_file.mean(1).mean(1))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255)
    for img_pth in M:
        img = np.array(Image.open(img_pth))
        net.blobs['data'].data[...] = transformer.preprocess('data', img / 255.0)
        # Forward Pass
        res = net.forward()
        # Stack the activations of feature maps to make local descriptors
        Features[layer] = np.array(net.blobs[layer].data[0].copy())
        StackedFeatures[layer] = Features[layer].transpose(1, 2, 0)
        # Retrieve N ROIs for test and ref images
        ROIs = getROIs(Features[layer], StackedFeatures[layer], img)
        vocabulary = vocab[N][V][layer]
        # Retrieve VLAD descriptors using ROIs and vocabulary
        VLAD = getVLAD(ROIs, vocabulary)
        ref_map_images_descs.append(VLAD)
    m_desc = np.stack(ref_map_images_descs).astype(np.float32)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, m_desc, type='map')
    return m_desc

def perform_vpr(q_path, M):
    return None
