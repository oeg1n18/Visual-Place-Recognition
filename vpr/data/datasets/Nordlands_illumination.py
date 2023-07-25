from glob import glob
import config
import csv
import numpy as np
from tqdm import tqdm

NAME = 'Nordlands_illumination'


def get_images_from_sections(pth):
    sections = sorted(glob(pth + '/*'))
    all_paths = []
    for sect in sections:
        all_paths += glob(sect + '/*')
    n_augs = max([int(pth[-5]) for pth in all_paths])
    all_queries = [[] for i in range(n_augs + 1)]
    for q in all_paths:
        all_queries[int(q[-5])].append(q)

    all_queries = [np.array(q) for q in all_queries]

    imgs = []
    for img_paths in all_queries:
        img_nums = np.array([int(p.split('/')[-1][:-9]) for p in img_paths])
        sort_idx = np.argsort(img_nums)
        imgs.append(np.array(img_paths)[sort_idx])
    imgs = np.array(imgs).flatten()
    return imgs


def get_map_images_from_sections(pth):
    imgs = []
    sections = sorted(glob(pth + '/*'))
    for sect in sections:
        img_paths = glob(sect + '/*')
        img_nums = np.array([int(p.split('/')[-1][:-4]) for p in img_paths])
        sort_idx = np.argsort(img_nums)
        imgs += list(np.array(img_paths)[sort_idx].flatten())
    return imgs



def get_query_paths(partition='test', query_set=["spring"]):
    assert partition in ["train", "test"]

    if partition == 'train':
        if query_set == "summer":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/summer_images_train_illumination'
            imgs = get_images_from_sections(pth)
            return imgs

        if query_set == "fall":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/fall_images_train_illumination'
            imgs = get_images_from_sections(pth)
            return imgs
        if query_set == "winter":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/winter_images_train_illumination'
            imgs = get_images_from_sections(pth)
            return imgs
        if query_set == "spring":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/spring_images_train_illumination'
            imgs = get_images_from_sections(pth)
            return imgs

        if type(query_set) == list:
            imgs = []
            for set in query_set:
                if set == "summer":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/summer_images_train_illumination'
                    imgs.append(get_images_from_sections(pth))
                if set == "winter":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/winter_images_train_illumination'
                    imgs.append(get_images_from_sections(pth))
                if set == "fall":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/fall_images_train_illumination'
                    imgs.append(get_images_from_sections(pth))
                if set == "spring":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/spring_images_train_illumination'
                    imgs.append(get_images_from_sections(pth))
                imgs = np.array(imgs).flatten()
            return imgs

    elif partition == 'test':
        if query_set == "summer":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/summer_images_test_illumination'
            imgs = get_images_from_sections(pth)
            return imgs
        if query_set == "fall":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/fall_images_test_illumination'
            imgs = get_images_from_sections(pth)
            return imgs
        if query_set == "winter":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/winter_images_test_illumination'
            imgs = get_images_from_sections(pth)
            return imgs
        if query_set == "spring":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/spring_images_test_illumination'
            imgs = get_images_from_sections(pth)
            return imgs
        if type(query_set) == list:
            imgs = []
            for set in query_set:
                if set == "summer":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/summer_images_test_illumination'
                    imgs.append(get_images_from_sections(pth))
                if set == "winter":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/winter_images_test_illumination'
                    imgs.append(get_images_from_sections(pth))
                if set == "fall":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/fall_images_test_illumination'
                    imgs.append(get_images_from_sections(pth))
                if set == "spring":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/spring_images_test_illumination'
                    imgs.append(get_images_from_sections(pth))
                imgs = np.array(imgs).flatten()
                return imgs
            return imgs


def get_map_paths(partition="test", reference_set=["summer"]):
    if partition == 'train':
        if reference_set == "summer":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/summer_images_train'
            imgs = get_map_images_from_sections(pth)
            return imgs
        if reference_set == "winter":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/winter_images_train'
            imgs = get_map_images_from_sections(pth)
            return imgs
        if reference_set == "fall":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/fall_images_train'
            imgs = get_map_images_from_sections(pth)
            return imgs
        if reference_set == "spring":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/spring_images_train'
            imgs = get_map_images_from_sections(pth)
            return imgs
        if type(reference_set) == list:
            imgs = []
            for set in reference_set:
                if set == "summer":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/summer_images_train'
                    imgs += get_map_images_from_sections(pth)
                if set == "winter":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/winter_images_train'
                    imgs += get_map_images_from_sections(pth)
                if set == "fall":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/fall_images_train'
                    imgs += get_map_images_from_sections(pth)
                if set == "spring":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/train/spring_images_train'
                    imgs += get_map_images_from_sections(pth)
            return imgs

    elif partition == 'test':
        if reference_set == "summer":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/summer_images_test'
            imgs = get_map_images_from_sections(pth)
            return imgs
        if reference_set == "winter":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/winter_images_test'
            imgs = get_map_images_from_sections(pth)
            return imgs
        if reference_set == "fall":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/fall_images_test'
            imgs = get_map_images_from_sections(pth)
            return imgs
        if reference_set == "spring":
            pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/spring_images_test'
            imgs = get_map_images_from_sections(pth)
            return imgs

        if type(reference_set) == list:
            imgs = []
            for set in reference_set:
                if set == "summer":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/summer_images_test'
                    imgs += get_map_images_from_sections(pth)
                if set == "winter":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/winter_images_test'
                    imgs += get_map_images_from_sections(pth)
                if set == "fall":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/fall_images_test'
                    imgs += get_map_images_from_sections(pth)
                if set == "spring":
                    pth = config.root_dir + '/vpr/data/raw_data/Nordlands_partitioned/test/spring_images_test'
                    imgs += get_map_images_from_sections(pth)
            return imgs


def get_gtmatrix(gt_type='soft', partition="test", query_set=["spring"], reference_set="summer"):
    Q = get_query_paths(partition=partition, query_set=query_set)
    M = get_map_paths(partition=partition, reference_set=reference_set)

    def soft_diag(size, softness=5):
        diag = np.eye(size).astype(np.uint8)
        for i in range(size):
            if i > softness < size - softness:
                diag[i, i - softness:i + softness] = 1
            if i < softness:
                diag[i, 0:i + softness] = 1
            if i >= size - softness:
                diag[i, i - softness:size] = 1
        return diag.astype(np.uint8)

    if len(Q) <= len(M):
        if gt_type == 'hard':
            GT = np.vstack([np.eye(len(Q)) for _ in range(int(len(M) / len(Q)))])
            return GT.astype(np.uint8)

        elif gt_type == 'soft':
            GT = np.vstack([soft_diag(len(Q)) for _ in range(int(len(M) / len(Q)))])
            return GT.astype(np.uint8)
    elif len(Q) > len(M):
        if gt_type == 'hard':
            GT = np.hstack([np.eye(len(M)) for _ in range(int(len(Q) / len(M)))])
            return GT.astype(np.uint8)

        elif gt_type == 'soft':
            GT = np.hstack([soft_diag(len(M)) for _ in range(int(len(Q) / len(M)))])
            return GT.astype(np.uint8)


Q = get_query_paths()
M = get_map_paths()
GT = get_gtmatrix()
