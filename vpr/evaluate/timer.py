from tqdm import tqdm

import config
import wandb
import time
import numpy as np
class Timer:
    def __init__(self, dataset, vpr_technique, sample_size=5, n_imgs=10, gpu=True):
        config.device = 'cuda' if gpu else 'cpu'
        self.dataset = dataset
        self.vpr = vpr_technique
        self.Q = dataset.get_query_paths(rootdir=config.root_dir)
        self.M = dataset.get_query_paths(rootdir=config.root_dir)
        GTsoft = dataset.get_gtmatrix(rootdir=config.root_dir, gt_type='soft')
        self.sample_size = sample_size
        self.n_imgs = n_imgs

        assert n_imgs <= len(self.Q), "Not enough query images for sample size"
        assert n_imgs <= len(self.M), "Not enough database images for sample size"

        wandb.login()
        self.run = wandb.init(
            project="VPR-Metrics",
            name=self.vpr.NAME,
            tags=[self.vpr.NAME, self.dataset.NAME, 'GTsoft' if isinstance(GTsoft, type(np.ones(1))) else 'GThard'],

            config={
                'method': self.vpr.NAME,
                'dataset': self.dataset.NAME,
                'GT_type': 'GTsoft' if isinstance(GTsoft, type(np.ones(1))) else 'GThard',
                'session_type': 'N/A'
            }
        )

    def run_evaluation(self):
        enc_mean, enc_var = self.encoding_time()
        match_mean, match_var = self.matching_time()

        wandb.run.summary["encoding_time_mean"] = enc_mean
        wandb.run.summary["encoding_time_var"] = enc_var
        wandb.run.summary["matching_time_mean"] = match_mean
        wandb.run.summary["matching_time_var"] = match_var

    def encoding_time(self):
        times = []
        pbar = tqdm(range(self.sample_size))
        for i in pbar:
            pbar.set_description("Computing Encoding Time")
            st = time.time()
            m_desc = self.vpr.compute_map_features(self.M[:self.n_imgs])
            et = time.time()
            times.append((et-st)/self.n_imgs)
        return np.mean(times), np.var(times)

    def matching_time(self):
        times = []
        m_desc = self.vpr.compute_map_features(self.M)
        pbar = tqdm(range(self.sample_size))
        for i in pbar:
            pbar.set_description("Computing Matching Time")
            st = time.time()
            for query in self.Q[:self.n_imgs]:
                _, _ = self.vpr.perform_vpr(query, m_desc)
            et = time.time()
            times.append((et-st)/self.n_imgs)
        return np.mean(times), np.var(times)

