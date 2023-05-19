import argparse


parser = argparse.ArgumentParser()


parser.add_argument('--mode', required=True, choices=("describe", "evaluate", "visualise"), help='Specify either describe or evaluate', type=str)
parser.add_argument('--dataset', choices=("Nordlands", "Pittsburgh", "ESSEX3IN1"), help='specify one of the datasets from src/data/raw_data', type=str)
parser.add_argument('--technique', choices=("netvlad", "densevlad"), help="specify one of the techniques from src/vpr_tecniques", type=str)
parser.add_argument('--language', choices=("python, cpp"), help="specify either python or cpp", type=str)

args = parser.parse_args()
