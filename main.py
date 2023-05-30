from vpr.data.datasets import GardensPointWalking, SFU, StLucia
from vpr.evaluate.metrics import Metrics
from vpr.vpr_techniques.utils import load_descriptors
import config
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--mode', required=True, choices=("describe", "eval_time", "eval_metrics", "eval_invariance"),
                    help='Specify either describe or evaluate', type=str)
parser.add_argument('--dataset', choices=("SFU", "GardensPointWalking", "StLucia"),
                    help='specify one of the datasets from vpr/data/raw_data', type=str, default="StLucia")
parser.add_argument('--method', choices=("patchNetVLAD", "denseVLAD", "MixVPR", "NetVLAD"),
                    help="specify one of the techniques from vpr/vpr_tecniques", type=str, default="patchNetVLAD")
args = parser.parse_args()

# ============== Chose the dataset ===============
if args.dataset == "StLucia":
    dataset = StLucia
elif args.dataset == "GardensPointWalking":
    dataset = GardensPointWalking
elif args.dataset == "SFU":
    dataset = SFU
else:
    dataset = GardensPointWalking

# ============= Chose the method ================
if args.method == "patchNetVLAD":
    from vpr.vpr_techniques import patchnetvlad
    method = patchnetvlad

elif args.method == "NetVLAD":
    from vpr.vpr_techniques import netvlad
    method = netvlad

elif args.method == "MixVPR":
    from vpr.vpr_techniques import mixvpr
    method = mixvpr
else:
    from vpr.vpr_techniques import patchnetvlad
    method = patchnetvlad


# =============== Describe Mode ==================
if args.mode == "describe":
    M = dataset.get_map_paths(rootdir=config.root_dir)
    Q = dataset.get_query_paths(rootdir=config.root_dir)
    GT = dataset.get_gtmatrix(rootdir=config.root_dir, gt_type='hard')
    GTsoft = dataset.get_gtmatrix(rootdir=config.root_dir, gt_type='soft')

    method.compute_query_desc(Q, dataset_name=dataset.NAME)
    method.compute_map_features(M, dataset_name=dataset.NAME)


# ============== Evaluate Mode =======================
if args.mode == "eval_metrics":
    M = dataset.get_map_paths(rootdir=config.root_dir)
    Q = dataset.get_query_paths(rootdir=config.root_dir)
    GT = dataset.get_gtmatrix(rootdir=config.root_dir, gt_type='hard')
    GTsoft = dataset.get_gtmatrix(rootdir=config.root_dir, gt_type='soft')

    Fq, Fm = load_descriptors(dataset.NAME, method.NAME)
    eval = Metrics(method.NAME, dataset.NAME, Fq, Fm, GT, q_pths=Q, db_pths=M, GTsoft=GTsoft, rootdir=config.root_dir)
    eval.log_metrics()


if args.mode == "eval_time":
    raise NotImplementedError

if args.mode == "eval_invariance":
    raise NotImplementedError
