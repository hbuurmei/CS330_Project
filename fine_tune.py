import argparse
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from load_data import TrajectoryDataset


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dynamics_model', help='What model to train')
parser.add_argument('--mode', default='last', help='Fine-tuning mode')  # {shared, head, all}
args = parser.parse_args()


# Fine tune function
def fine_tune(model, mode):
    # Define which parameters to tune
    num_shared = model.num_shared
    if mode == 'shared':
        ft_params = 
    elif mode == 'head':
        ft_params = 
    elif mode == 'all':
        ft_params = list(model.parameters())
    else:
        raise ValueError(f"Unrecognized fine-tuning mode {mode}")
