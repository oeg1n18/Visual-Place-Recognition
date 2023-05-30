from vpr.data.datasets import GardensPointWalking, SFU, StLucia
from vpr.vpr_techniques import mixvpr, patchnetvlad, netvlad
from vpr.evaluate.metrics import Metrics
import config

'''
parser = argparse.ArgumentParser()

parser.add_argument('--mode', required=True, choices=("describe", "eval_time", "eval_metrics", "eval_invariance"),
                    help='Specify either describe or evaluate', type=str)
parser.add_argument('--dataset', choices=("SFU", "GardensPointWalking", "StLucia"),
                    help='specify one of the datasets from vpr/data/raw_data', type=str, default="StLucia")
parser.add_argument('--technique', choices=("patchNetVLAD", "denseVLAD", "mixvpr", "NetVLAD"),
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
if args.technique == "patchNetVLAD":
    method = patchnetvlad
elif args.technique == "denseVLAD":
    method = densevlad
elif args.technique == "NetVLAD":
    method = netvlad
elif args.technqiue == "mixvpr":
    method = mixvpr
else:
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

'''
dataset = GardensPointWalking
M = dataset.get_map_paths(rootdir=config.root_dir)
Q = dataset.get_query_paths(rootdir=config.root_dir)
GT = dataset.get_gtmatrix(rootdir=config.root_dir, gt_type='hard')
GTsoft = dataset.get_gtmatrix(rootdir=config.root_dir, gt_type='soft')


Fq = mixvpr.compute_query_desc(Q)
Fm = mixvpr.compute_map_features(M)

eval = Metrics(mixvpr.NAME, dataset.NAME, Fq, Fm, GT, q_pths=Q, db_pths=M, GTsoft=GTsoft, rootdir=config.root_dir)
eval.log_metrics()


Fq = patchnetvlad.compute_query_desc(Q)
Fm = patchnetvlad.compute_map_features(M)

eval = Metrics(patchnetvlad.NAME, dataset.NAME, Fq, Fm, GT, q_pths=Q, db_pths=M, GTsoft=GTsoft,
               matching_method=patchnetvlad.matching_function, rootdir=config.root_dir)
eval.log_metrics()