import torch
import src.data.utils as dataset
import src.python.vpr.resnet.vpr as resnet_vpr
from src.python.evaluate.vpr_eval import Evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the Data Module
dm = dataset.DataModule(dataset='GardensPointWalking', session_type='ms', gt_type='hard')

# Load the VPR method
vpr = resnet_vpr.VPR(device=device, batch_size=32)

# Instantiate the evaluation module
eval = Evaluate(vpr, dm)

# Evaluate the performance
eval.profile(threshold=0.7, p=1., k=1000, N=10)


